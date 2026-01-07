# new_step4_single_trainer.py
# Step4: 损失函数（Mixture NLL，多专家轨迹分布） + Trainer
# 适配：Step3 模型 forward() 返回 dict:
#   out = {
#     "logits": (B,K),
#     "mu": (B,K,P,2)   # 每步位移均值（RevIN norm空间）
#     "sigma": (B,K,P,2)# 每步 std（RevIN norm空间，>0）
#     "rho": (B,K,P,1)  # 相关系数
#     "aux_loss": scalar
#     "r_mean": (B,1,2) # RevIN mean (meters)
#     "r_std":  (B,1,2) # RevIN std  (meters)
#     "expert_usage": (E,) or None
#   }

import os
import logging
import math
from typing import Dict, Any, Optional, Tuple
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Trainer")


# ==========================================================
# 1) Utils
# ==========================================================
def _bivar_gaussian_nll_meters(
    mu_m: torch.Tensor,
    sigma_m: torch.Tensor,
    rho: torch.Tensor,
    y_m: torch.Tensor,
) -> torch.Tensor:
    """
    二维高斯 NLL（单位：meters 空间）
    mu_m:    (B,K,P,2)
    sigma_m: (B,K,P,2) positive
    rho:     (B,K,P,1) in [-1,1]
    y_m:     (B,1,P,2) or (B,K,P,2)

    return:
      nll_k: (B,K)  (对时间 P 求 mean，数值更稳)
    """
    
    mu_m = mu_m.to(dtype=torch.float32)
    sigma_m = sigma_m.to(dtype=torch.float32)
    rho = rho.to(dtype=torch.float32)
    y_m = y_m.to(dtype=torch.float32)

    sx = sigma_m[..., 0].clamp_min(1e-6)
    sy = sigma_m[..., 1].clamp_min(1e-6)
    r = rho[..., 0].clamp(-0.999, 0.999)

    zx = (y_m[..., 0] - mu_m[..., 0]) / sx
    zy = (y_m[..., 1] - mu_m[..., 1]) / sy

    one = (1.0 - r * r).clamp_min(1e-6)

    # log normalization
    log_norm = torch.log(sx) + torch.log(sy) + 0.5 * torch.log(one) + math.log(2.0 * math.pi)

    quad = (zx * zx + zy * zy - 2.0 * r * zx * zy) / one
    nll_t = log_norm + 0.5 * quad  # (B,K,P)

    # time mean: (B,K)
    return nll_t.mean(dim=-1)
    # 整条未来轨迹总体上有多合理（每个 expert 一个总分）
    # 对每步算完后，再把所有步平均

def _bivar_gaussian_nll_meters_per_t(
    mu_m: torch.Tensor,
    sigma_m: torch.Tensor,
    rho: torch.Tensor,
    y_m: torch.Tensor,
) -> torch.Tensor:
    """
    与 _bivar_gaussian_nll_meters 相同，但返回每个时间步的 NLL：
      mu_m/sigma_m/rho: (B,K,P,*)
      y_m:             (B,1,P,2) or (B,K,P,2)
    返回: (B,K,P)
    """
    mu_m = mu_m.to(dtype=torch.float32)
    sigma_m = sigma_m.to(dtype=torch.float32)
    rho = rho.to(dtype=torch.float32)
    y_m = y_m.to(dtype=torch.float32)

    sx = sigma_m[..., 0].clamp_min(1e-6)
    sy = sigma_m[..., 1].clamp_min(1e-6)
    r = rho[..., 0].clamp(-0.999, 0.999)

    zx = (y_m[..., 0] - mu_m[..., 0]) / sx
    zy = (y_m[..., 1] - mu_m[..., 1]) / sy

    one = (1.0 - r * r).clamp_min(1e-6)
    log_norm = torch.log(sx) + torch.log(sy) + 0.5 * torch.log(one) + math.log(2.0 * math.pi)

    quad = (zx * zx + zy * zy - 2.0 * r * zx * zy) / one
    return log_norm + 0.5 * quad  # (B,K,P)
    # per-step gate（因为 gate 也是每步一套概率分布）
    # Viterbi path（每步每专家都有打分）


def _pairwise_endpoint_diversity(end_xy: torch.Tensor, scale_m: float = 50.0) -> torch.Tensor:
    """
    end_xy: (B,K,2)  每个 mode 的终点偏移（meters）
    返回: (B,) diversity loss
    设计：惩罚 mode 终点彼此太近（exp(-d/scale)）
    """
    B, K, _ = end_xy.shape
    diff = end_xy.unsqueeze(2) - end_xy.unsqueeze(1)        # (B,K,K,2)
    dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-9)     # (B,K,K)

    # exp(-d/s) 越近越大 -> 惩罚越大
    div = torch.exp(-dist / float(scale_m))                 # (B,K,K)

    # 去除对角线（自己和自己）
    div_sum = div.sum(dim=(1, 2)) - div.diagonal(dim1=1, dim2=2).sum(dim=1)
    denom = float(K * K - K) + 1e-6
    return div_sum / denom


# ==========================================================
# 1.1) 受限切换 Viterbi（用于“单条轨迹”的路径对齐）
# ==========================================================
def _viterbi_scores_limited_switch(score_tk: np.ndarray, max_switches: int, switch_cost: float) -> np.ndarray:
    """
    score_tk: (P,K) 每步每个专家的得分（越大越好），例如 log_pi - nll
    max_switches: 允许的最大切换次数
    switch_cost:  每次切换的惩罚（越大越不切换）
    return: k_steps (P,) int64
    """
    score_tk = np.asarray(score_tk, dtype=np.float64)  # (P,K)
    if score_tk.ndim != 2:
        raise ValueError(f"score_tk must be (P,K), got {score_tk.shape}")
    P, K = score_tk.shape
    S = int(max(0, max_switches))  # 最大切换次数

    # dp[t, k, s] = 到时间步 t 为止（包含 t），最后一步选专家 k，并且已经发生了 s 次切换时，能得到的 最大累计总分。
    dp = np.full((P, K, S + 1), -np.inf, dtype=np.float64)
    # prev_k[t, k, s] = 这个最优值是上一步选的那个专家 k
    prev_k = np.full((P, K, S + 1), -1, dtype=np.int16)
    # prev_s[t, k, s] = 这个最优值是上一步用了多少次切换 s
    prev_s = np.full((P, K, S + 1), -1, dtype=np.int16)

    dp[0, :, 0] = score_tk[0]
    prev_k[0, :, 0] = -1
    prev_s[0, :, 0] = -1

    for t in range(1, P):
        for s in range(S + 1):
            # stay
            dp[t, :, s] = dp[t - 1, :, s] + score_tk[t]
            prev_k[t, :, s] = np.arange(K, dtype=np.int16)
            prev_s[t, :, s] = s

            if s == 0 or K <= 1:
                continue

            # switch: from k_prev != k
            base = dp[t - 1, :, s - 1] - float(switch_cost)  # (K,)
            for k in range(K):
                # best prev excluding k
                cand = base.copy()
                cand[k] = -np.inf
                k_prev = int(np.argmax(cand))
                if not np.isfinite(cand[k_prev]):
                    continue
                sc = float(cand[k_prev] + score_tk[t, k]) # 切换情况下的总分(sc=上一时刻 - 切换惩罚 + 当前步得分)
                if sc > dp[t, k, s]:
                    dp[t, k, s] = sc
                    prev_k[t, k, s] = k_prev
                    prev_s[t, k, s] = s - 1

    flat = int(np.argmax(dp[P - 1]))
    end_k, end_s = np.unravel_index(flat, (K, S + 1))

    ks = np.zeros((P,), dtype=np.int64)
    k = int(end_k)
    s = int(end_s)
    for t in range(P - 1, -1, -1):
        ks[t] = k
        if t == 0:
            break
        pk = int(prev_k[t, k, s])
        ps = int(prev_s[t, k, s])
        k, s = pk, ps
    return ks


def _count_switches_np(k_steps: np.ndarray) -> int:
    k = np.asarray(k_steps).astype(np.int64).reshape(-1)
    if k.size <= 1:
        return 0
    return int(np.sum(k[1:] != k[:-1]))


def _viterbi_scores_limited_switch_torch(
    score: torch.Tensor, max_switches: int, switch_cost: float
) -> torch.Tensor:
    """
    Torch 版受限切换 Viterbi（用于训练阶段加速，避免 score->CPU->numpy 的同步开销）。

    score: (B,P,K) 每步每个专家的得分（越大越好），例如 log_pi - nll
    返回: k_steps (B,P) long
    """
    if score.ndim != 3:
        raise ValueError(f"score must be (B,P,K), got {tuple(score.shape)}")

    score = score.float()
    B = int(score.shape[0])
    P = int(score.shape[1])
    K = int(score.shape[2])
    S = int(max(0, max_switches))
    device = score.device
    dtype = score.dtype
    neg_inf = torch.tensor(-float("inf"), device=device, dtype=dtype) #保证 Viterbi 只会选择合法路径，不会被无效状态干扰

    # dp[b,k,s] = best score up to time t, ending at expert k with s switches used.
    dp = torch.full((B, K, S + 1), neg_inf, device=device, dtype=dtype)
    prev_k = torch.full((P, B, K, S + 1), -1, device=device, dtype=torch.int16)
    prev_s = torch.full((P, B, K, S + 1), -1, device=device, dtype=torch.int16)

    dp[:, :, 0] = score[:, 0, :]

    k_grid = torch.arange(K, device=device, dtype=torch.long).view(1, K).expand(B, K)  # (B,K)

    for t in range(1, P):
        st = score[:, t, :]  # (B,K)
        dp_prev = dp
        dp_next = torch.full((B, K, S + 1), neg_inf, device=device, dtype=dtype)

        for s in range(0, S + 1):
            stay_score = dp_prev[:, :, s] + st  # (B,K)
            best_score = stay_score
            best_pk = k_grid
            best_ps = torch.full((B, K), s, device=device, dtype=torch.long)

            if s > 0 and K > 1:
                prev = dp_prev[:, :, s - 1]  # (B,K)

                # best and 2nd-best previous expert per batch (to get best excluding k)
                best_vals, best_idx = torch.topk(prev, k=2, dim=1)
                best1_val, best2_val = best_vals[:, 0], best_vals[:, 1]
                best1_idx, best2_idx = best_idx[:, 0], best_idx[:, 1]

                best1_idx_b = best1_idx.view(B, 1).expand(B, K)
                best2_idx_b = best2_idx.view(B, 1).expand(B, K)
                best1_val_b = best1_val.view(B, 1).expand(B, K)
                best2_val_b = best2_val.view(B, 1).expand(B, K)

                same_as_best1 = best1_idx_b == k_grid
                best_other_val = torch.where(same_as_best1, best2_val_b, best1_val_b)
                best_other_idx = torch.where(same_as_best1, best2_idx_b, best1_idx_b)

                switch_score = best_other_val - float(switch_cost) + st
                use_switch = switch_score > stay_score

                best_score = torch.where(use_switch, switch_score, stay_score)
                best_pk = torch.where(use_switch, best_other_idx, k_grid)
                best_ps = torch.where(use_switch, torch.full_like(best_ps, s - 1), best_ps)

            dp_next[:, :, s] = best_score
            prev_k[t, :, :, s] = best_pk.to(torch.int16)
            prev_s[t, :, :, s] = best_ps.to(torch.int16)

        dp = dp_next

    # pick best terminal state
    flat = torch.argmax(dp.reshape(B, -1), dim=1)  # (B,)
    end_k = (flat // (S + 1)).to(torch.long)
    end_s = (flat % (S + 1)).to(torch.long)

    ks = torch.zeros((B, P), device=device, dtype=torch.long)
    k = end_k
    s = end_s
    b_idx = torch.arange(B, device=device, dtype=torch.long)

    for t in range(P - 1, -1, -1):
        ks[:, t] = k
        if t == 0:
            break
        pk = prev_k[t, b_idx, k, s].to(torch.long)
        ps = prev_s[t, b_idx, k, s].to(torch.long)
        k, s = pk, ps

    return ks  # k_steps (B,P)每个样本在每个未来步选到的专家索引序列


# ==========================================================
# 2) SOTA Loss: Mixture NLL + Soft-best-of-K + Diversity + Entropy
# ==========================================================
class MixtureTrajectoryLoss(nn.Module):
    """
    强且稳的 SOTA 方向：
      - 主：Mixture NLL（多专家轨迹分布）
      - 辅：soft-best-of-K regression（防止所有 mode 都学不好）
      - 防塌缩：diversity（mode 终点分散）
      - 稳定：entropy（极小权重，早期防 mode collapse）

    loss_cfg 建议（你 config 里）：
    {
      "nll": 1.0,
      "soft_bestofk": 0.15,
      "diversity": 0.02,
      "entropy": 0.001,
      "softmin_tau": 0.5,
      "min_sigma_m": 0.8
    }
    """
    def __init__(self, loss_cfg: Dict[str, Any]):
        super().__init__()
        self.w_nll = float(loss_cfg.get("nll", 1.0))
        self.w_best = float(loss_cfg.get("soft_bestofk", 0.15))
        self.w_div = float(loss_cfg.get("diversity", 0.02))
        self.w_ent = float(loss_cfg.get("entropy", 0.001))
        self.entropy_start = float(loss_cfg.get("entropy", self.w_ent))
        self.entropy_decay_epochs = int(loss_cfg.get("entropy_decay_epochs", 0))

        self.tau = float(loss_cfg.get("softmin_tau", 0.5))
        self.min_sigma_m = float(loss_cfg.get("min_sigma_m", 0.8))

        # per-step gate 正则：抑制频繁切换 + 防塌缩（鼓励专家都有机会被使用）
        self.w_switch = float(loss_cfg.get("switch", 0.0))
        self.w_lb = float(loss_cfg.get("load_balance", 0.0))

        # diversity 距离尺度（不写到 config 也行，默认 50m）
        self.div_scale_m = float(loss_cfg.get("div_scale_m", 50.0))

    def maybe_update_entropy(self, epoch: int) -> None:
        """
        可选：entropy 权重衰减（前期防 mode collapse，后期让 gate 变尖更利于 top-k 输出）。
        epoch 从 0 开始。
        """
        if self.entropy_decay_epochs <= 0:
            return
        e = int(max(epoch, 0))
        if e >= self.entropy_decay_epochs:
            self.w_ent = 0.0
        else:
            ratio = 1.0 - (float(e) / float(self.entropy_decay_epochs))
            self.w_ent = float(self.entropy_start) * max(ratio, 0.0)

    @staticmethod
    def _softmin(values: torch.Tensor, tau: float) -> torch.Tensor:
        """
        values: (B,K) -> softmin (B,)
        """
        return -tau * torch.logsumexp(-values / tau, dim=-1)

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        y_target_m: torch.Tensor,   # (B,P,2) meters/step
        x_hist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # gate：优先用 per-step logits_step(B,P,K)，否则退化为全局 logits(B,K)
        logits_global = out["logits"]  # (B,K)
        logits_step = out.get("logits_step", None)  # (B,P,K) or None
        mu_n = out["mu"]               # (B,K,P,2) norm
        sigma_n = out["sigma"]         # (B,K,P,2) norm
        rho = out["rho"]               # (B,K,P,1)

        r_mean = out["r_mean"]         # (B,1,2) meters
        r_std = out["r_std"]           # (B,1,2) meters

        # 反归一化到 meters/step
        mu_m = mu_n * r_std.unsqueeze(1) + r_mean.unsqueeze(1)   # (B,K,P,2)
        sigma_m = sigma_n * r_std.unsqueeze(1)                   # (B,K,P,2)
        # AMP 下统一用 float32 算 loss，避免溢出导致 NaN
        mu_m = mu_m.to(dtype=torch.float32)
        sigma_m = sigma_m.to(dtype=torch.float32)
        rho = rho.to(dtype=torch.float32)

        # sigma 下限（meters）：强烈建议，避免 NLL 数值爆炸/不稳
        sigma_m = sigma_m.clamp_min(self.min_sigma_m)

        # ========== 1) Mixture NLL（per-step gate）==========
        # y_m -> (B,1,P,2)
        y_m = y_target_m.unsqueeze(1).to(dtype=torch.float32)
        nll_k_t = _bivar_gaussian_nll_meters_per_t(mu_m, sigma_m, rho, y_m)  # (B,K,P)

        if logits_step is None:
            log_pi_k = F.log_softmax(logits_global, dim=-1).to(dtype=torch.float32)  # (B,K)
            log_pi_k_t = log_pi_k.unsqueeze(-1).expand_as(nll_k_t)           # (B,K,P)
        else:
            # (B,P,K) -> (B,K,P)
            log_pi_k_t = F.log_softmax(logits_step, dim=-1).permute(0, 2, 1).to(dtype=torch.float32)  # (B,K,P)

        mix_nll_t = -torch.logsumexp(log_pi_k_t - nll_k_t, dim=1)            # (B,P)
        mix_nll = mix_nll_t.mean(dim=-1)                                     # (B,)

        # ========== 2) Soft-best-of-K (NLL softmin) ==========
        # 用 NLL 做 softmin 比 MSE 更稳定/更一致（尤其当你最终评估也是基于高斯分布 NLL/热力图）。
        nll_k = nll_k_t.mean(dim=-1)                                    # (B,K)
        best_soft = self._softmin(nll_k, self.tau)                      # (B,)

        # ========== 3) Diversity (endpoint) ==========
        end_xy = mu_m.sum(dim=2)                                        # (B,K,2) endpoint offset (meters)
        div_loss = _pairwise_endpoint_diversity(end_xy, scale_m=self.div_scale_m)  # (B,)

        # ========== 4) Entropy（可选；建议衰减到 0，避免 gate 长期平均分配）==========
        if logits_step is None:
            pi = torch.softmax(logits_global, dim=-1)                         # (B,K)
            entropy = -(pi * (pi + 1e-9).log()).sum(dim=-1)                   # (B,)
        else:
            pi_step = torch.softmax(logits_step, dim=-1)                      # (B,P,K)
            entropy = -(pi_step * (pi_step + 1e-9).log()).sum(dim=-1).mean(dim=-1)  # (B,)
        ent_loss = -entropy                                              # maximize entropy => minimize -entropy

        # ========== 5) Switching regularizer（抑制每步频繁切换专家）==========
        if (logits_step is None) or (self.w_switch <= 0.0):
            sw_loss = torch.zeros_like(mix_nll)
        else:
            pi_step = torch.softmax(logits_step, dim=-1)                      # (B,P,K)
            sim = (pi_step[:, 1:, :] * pi_step[:, :-1, :]).sum(dim=-1)        # (B,P-1)
            sw_loss = (1.0 - sim).mean(dim=-1)                                # (B,)

        # ========== 6) Load balance（防塌缩：让专家都有机会被用到）==========
        if (logits_step is None) or (self.w_lb <= 0.0):
            lb_loss = torch.zeros_like(mix_nll)
        else:
            pi_step = torch.softmax(logits_step, dim=-1)                      # (B,P,K)
            usage = pi_step.mean(dim=(0, 1))                                  # (K,)
            target = torch.full_like(usage, 1.0 / float(usage.numel()))
            lb_loss = ((usage - target) ** 2).mean().expand_as(mix_nll)       # (B,)

        # ========== 7) Reduce ==========
        L_nll = mix_nll.mean()
        L_best = best_soft.mean()
        L_div = div_loss.mean()
        L_ent = ent_loss.mean()
        L_sw = sw_loss.mean()
        L_lb = lb_loss.mean()

        total = (
            self.w_nll * L_nll
            + self.w_best * L_best
            + self.w_div * L_div
            + self.w_ent * L_ent
            + self.w_switch * L_sw
            + self.w_lb * L_lb
        )

        # 额外统计：top1 usage / minADE 上界可以在 validate 里算
        log = {
            "mix_nll": float(L_nll.detach().cpu()),
            "best_soft": float(L_best.detach().cpu()),
            "div": float(L_div.detach().cpu()),
            "entropy": float(entropy.mean().detach().cpu()),
            "switch": float(L_sw.detach().cpu()),
            "lb": float(L_lb.detach().cpu()),
        }
        return total, log


# ==========================================================
# 2.2) 单条轨迹：Viterbi 路径对齐损失（更贴近部署）
# ==========================================================
class SingleTrajectoryPathLoss(nn.Module):
    """
    目标：训练 K 个专家，但最终只输出一条“分段切换”的轨迹。

    思路（硬对齐 / 类 EM）：
    1) 对每个样本，计算每步每专家的 posterior score：
         score(t,k) = log_pi(t,k) - nll(t,k)
    2) 用受限切换 Viterbi 选出最优专家序列 k_t（单条路径）
    3) 只回传该路径上的 NLL（让专家学会），并用 CE 监督 gate（让 gate 学会输出该路径）
    """

    def __init__(self, loss_cfg: Dict[str, Any]):
        super().__init__()
        self.w_path = float(loss_cfg.get("path_nll", 1.0))
        self.w_gate = float(loss_cfg.get("path_gate_ce", 0.1))
        self.w_lb = float(loss_cfg.get("path_load_balance", loss_cfg.get("load_balance", 0.0)))
        self.w_ent = float(loss_cfg.get("path_entropy", loss_cfg.get("entropy", 0.0)))
        self.entropy_start = float(loss_cfg.get("path_entropy", loss_cfg.get("entropy", self.w_ent)))
        self.entropy_decay_epochs = int(loss_cfg.get("path_entropy_decay_epochs", 0))
        self.w_smooth = float(loss_cfg.get("path_smooth_lam", 0.0))
        self.min_sigma_m = float(loss_cfg.get("min_sigma_m", 0.8))
        self.max_switches = int(loss_cfg.get("path_max_switches", 2))
        self.switch_cost = float(loss_cfg.get("path_switch_cost", 1.0))

        # gate_ce 预热：前 N 轮从 0 线性升到目标值，避免早期 gate 乱导致对齐不稳
        self.gate_warmup_epochs = int(loss_cfg.get("path_gate_warmup_epochs", 3))
        self._gate_scale = 1.0

    def maybe_update_epoch(self, epoch: int) -> None:
        if self.gate_warmup_epochs <= 0:
            self._gate_scale = 1.0
            return
        e = int(max(epoch, 0))
        self._gate_scale = float(min(1.0, (e + 1) / float(self.gate_warmup_epochs)))

    def maybe_update_entropy(self, epoch: int) -> None:
        if self.entropy_decay_epochs <= 0:
            return
        e = int(max(epoch, 0))
        if e >= self.entropy_decay_epochs:
            self.w_ent = 0.0
        else:
            ratio = 1.0 - (float(e) / float(self.entropy_decay_epochs))
            self.w_ent = float(self.entropy_start) * max(ratio, 0.0)

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        y_target_m: torch.Tensor,   # (B,P,2) meters/step
        x_hist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        logits_step = out.get("logits_step", None)
        if logits_step is None:
            raise RuntimeError("SingleTrajectoryPathLoss requires model output 'logits_step' (B,P,K).")

        # float32 for stability
        logits_step = logits_step.float()           # (B,P,K) per-step gate logits
        mu_n = out["mu"].float()                    # (B,K,P,2) 每个专家、每步的位移均值（米/步
        sigma_n = out["sigma"].float()              # (B,K,P,2) 每个专家、每步的标准差（米/步）
        rho = out["rho"].float()                    # (B,K,P,1)
        r_mean = out["r_mean"].float()              # (B,1,2)
        r_std = out["r_std"].float()                # (B,1,2)
        y_target_m = y_target_m.float()             # (B,P,2)

        # meters/step
        mu_m = mu_n * r_std.unsqueeze(1) + r_mean.unsqueeze(1)
        sigma_m = (sigma_n * r_std.unsqueeze(1)).clamp_min(self.min_sigma_m)
        y_m = y_target_m.unsqueeze(1)               # (B,1,P,2)

        # (B,K,P)
        nll_k_t = _bivar_gaussian_nll_meters_per_t(mu_m, sigma_m, rho, y_m)
        # (B,P,K)
        log_pi = F.log_softmax(logits_step, dim=-1)  # gate 在第 t 步给专家 k 的对数概率
        score = (log_pi - nll_k_t.permute(0, 2, 1))  # (B,P,K) 这是每一步专家的权重

        # decode (torch, avoid CPU sync)
        B, P, K = score.shape
        with torch.no_grad():
            # 第 b 条样本在未来第 t 步选哪个 expert
            k_steps = _viterbi_scores_limited_switch_torch(score, self.max_switches, self.switch_cost)  # (B,P)

        # selected NLL along path
        nll_tk = nll_k_t.permute(0, 2, 1)  # (B,P,K)
        nll_sel = nll_tk.gather(2, k_steps.unsqueeze(-1)).squeeze(-1)  # (B,P) 这一步选中的专家的 NLL
        path_nll = nll_sel.mean(dim=-1)  # (B,) 这就是“专家拟合 GT 的主损失”

        # gate CE 多分类监督学习
        ce = F.cross_entropy(logits_step.reshape(B * P, K), k_steps.reshape(B * P), reduction="none")
        gate_ce = ce.view(B, P).mean(dim=-1)  # (B,) 什么时候该用哪个专家

        # optional entropy/load-balance/smoothness
        if self.w_ent > 0.0:
            pi_step = torch.softmax(logits_step, dim=-1)  # (B,P,K)
            entropy = -(pi_step * (pi_step + 1e-9).log()).sum(dim=-1).mean(dim=1)  # (B,)
            ent_loss = -entropy
        else:
            pi_step = None
            entropy = torch.zeros_like(path_nll)
            ent_loss = torch.zeros_like(path_nll)

        if self.w_lb > 0.0:
            if pi_step is None:
                pi_step = torch.softmax(logits_step, dim=-1)
            usage = pi_step.mean(dim=(0, 1))  # (K,)
            target = torch.full_like(usage, 1.0 / float(usage.numel()))
            lb_loss = ((usage - target) ** 2).mean().expand_as(path_nll)
        else:
            lb_loss = torch.zeros_like(path_nll)

        if self.w_smooth > 0.0:
            mu_bpk2 = mu_m.permute(0, 2, 1, 3)  # (B,P,K,2)
            mu_sel = mu_bpk2.gather(2, k_steps[:, :, None, None].expand(-1, -1, 1, 2)).squeeze(2)  # (B,P,2)
            if mu_sel.size(1) > 1:
                dv = mu_sel[:, 1:] - mu_sel[:, :-1]
                smooth = (dv ** 2).sum(dim=-1).mean(dim=-1)  # (B,)
            else:
                smooth = torch.zeros_like(path_nll)
        else:
            smooth = torch.zeros_like(path_nll)

        # reduce
        L_path = path_nll.mean()
        L_gate = gate_ce.mean()
        L_ent = ent_loss.mean()
        L_lb = lb_loss.mean()
        L_smooth = smooth.mean()

        total = (
            self.w_path * L_path
            + (self.w_gate * self._gate_scale) * L_gate
            + self.w_ent * L_ent
            + self.w_lb * L_lb
            + self.w_smooth * L_smooth
        )

        log = {
            "path_nll": float(L_path.detach().cpu()),
            "gate_ce": float(L_gate.detach().cpu()),
            "entropy": float(entropy.mean().detach().cpu()),
            "lb": float(L_lb.detach().cpu()),
            "smooth": float(L_smooth.detach().cpu()),
        }
        return total, log


# ==========================================================
# 2.1) Soft-best-of-K (NLL) + Gate 对齐 + Diversity + Entropy
# ==========================================================
class SoftBestKTrajectoryLoss(nn.Module):
    """
    方案 S+（更适合“输出 Top-K 轨迹 + 热力图”）：
      - 主：soft-best-of-K（用每个 mode 的高斯 NLL，而不是 MSE）
      - 让 gate 有意义：用 per-mode NLL 生成软标签 q，对齐 gate 分布（避免 top-k 选择不稳定）
      - 防塌缩：diversity（mode 终点分散）
      - 稳定：entropy（极小权重，早期防 mode collapse）

    你不需要把 gate 当作“严格概率”，但它必须能稳定选出 Top-K modes。

    loss_cfg 推荐：
    {
      "best_nll": 1.0,          # soft-best-of-K NLL 权重（主）
      "gate_ce": 0.1,           # gate 对齐权重（让 logits 有监督信号）
      "diversity": 0.02,
      "entropy": 0.001,
      "softmin_tau": 0.5,       # soft-best-of-K 温度（越小越接近 hard best）
      "tau_gate": 1.0,          # 生成 q 的温度（越小越尖锐）
      "min_sigma_m": 0.8
    }
    """
    def __init__(self, loss_cfg: Dict[str, Any]):
        super().__init__()
        self.w_best = float(loss_cfg.get("best_nll", loss_cfg.get("soft_bestofk", 1.0)))
        self.w_gate = float(loss_cfg.get("gate_ce", 0.1))
        self.w_div = float(loss_cfg.get("diversity", 0.02))
        self.w_ent = float(loss_cfg.get("entropy", 0.001))
        self.entropy_start = float(loss_cfg.get("entropy", self.w_ent))
        self.entropy_decay_epochs = int(loss_cfg.get("entropy_decay_epochs", 0))

        self.tau = float(loss_cfg.get("softmin_tau", 0.5))
        self.tau_gate = float(loss_cfg.get("tau_gate", 1.0))
        self.min_sigma_m = float(loss_cfg.get("min_sigma_m", 0.8))
        self.div_scale_m = float(loss_cfg.get("div_scale_m", 50.0))

    def maybe_update_entropy(self, epoch: int) -> None:
        if self.entropy_decay_epochs <= 0:
            return
        e = int(max(epoch, 0))
        if e >= self.entropy_decay_epochs:
            self.w_ent = 0.0
        else:
            ratio = 1.0 - (float(e) / float(self.entropy_decay_epochs))
            self.w_ent = float(self.entropy_start) * max(ratio, 0.0)

    @staticmethod
    def _softmin(values: torch.Tensor, tau: float) -> torch.Tensor:
        return -tau * torch.logsumexp(-values / tau, dim=-1)

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        y_target_m: torch.Tensor,   # (B,P,2) meters/step
        x_hist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # gate：优先用 per-step logits_step(B,P,K)，否则退化为全局 logits(B,K)
        logits_global = out["logits"]  # (B,K)
        logits_step = out.get("logits_step", None)  # (B,P,K) or None
        mu_n = out["mu"]               # (B,K,P,2) norm
        sigma_n = out["sigma"]         # (B,K,P,2) norm
        rho = out["rho"]               # (B,K,P,1)

        r_mean = out["r_mean"]         # (B,1,2) meters
        r_std = out["r_std"]           # (B,1,2) meters

        # 反归一化到 meters/step
        mu_m = mu_n * r_std.unsqueeze(1) + r_mean.unsqueeze(1)   # (B,K,P,2)
        sigma_m = sigma_n * r_std.unsqueeze(1)                   # (B,K,P,2)
        sigma_m = sigma_m.clamp_min(self.min_sigma_m)

        y_m = y_target_m.unsqueeze(1)                             # (B,1,P,2)
        nll_k = _bivar_gaussian_nll_meters(mu_m, sigma_m, rho, y_m)  # (B,K)

        # ========== 1) Soft-best-of-K NLL ==========
        best_soft = self._softmin(nll_k, self.tau)                # (B,)

        # ========== 2) Gate 对齐（soft label from NLL） ==========
        # q_k ∝ exp(-nll_k / tau_gate)
        q = torch.softmax(-nll_k / max(self.tau_gate, 1e-6), dim=-1)       # (B,K)
        log_pi = F.log_softmax(logits_global, dim=-1)                      # (B,K)
        gate_ce = -(q * log_pi).sum(dim=-1)                                # (B,)

        # ========== 3) Diversity ==========
        end_xy = mu_m.sum(dim=2)                                           # (B,K,2)
        div_loss = _pairwise_endpoint_diversity(end_xy, scale_m=self.div_scale_m)  # (B,)

        # ========== 4) Entropy（防塌缩，可很小） ==========
        pi = torch.softmax(logits_global, dim=-1)
        entropy = -(pi * (pi + 1e-9).log()).sum(dim=-1)                    # (B,)
        ent_loss = -entropy

        # ========== 5) Reduce ==========
        L_best = best_soft.mean()
        L_gate = gate_ce.mean()
        L_div = div_loss.mean()
        L_ent = ent_loss.mean()
        ent_mean = entropy.mean()
        min_nll = nll_k.min(dim=-1).values.mean()

        total = self.w_best * L_best + self.w_gate * L_gate + self.w_div * L_div + self.w_ent * L_ent
        log = {
            "best_nll": float(L_best.detach().cpu()),
            "gate_ce": float(L_gate.detach().cpu()),
            "div": float(L_div.detach().cpu()),
            "entropy": float(ent_mean.detach().cpu()),
            "min_nll": float(min_nll.detach().cpu()),
        }
        return total, log


# ==========================================================
# 3) Trainer (AMP + Clip + Save/Load + Metrics)
# ==========================================================
class Trainer:
    def __init__(
        self, train_loader, val_loader, optimizer, scheduler,
        model: nn.Module,
        criterion: nn.Module,
        device: str,
        save_dir: str,
        use_amp: bool = True,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.use_amp = bool(use_amp)
        self.grad_clip = float(grad_clip)

        self._amp_enabled = self.use_amp and device.startswith("cuda")

        # torch>=2.0 使用 torch.amp
        from torch.amp.grad_scaler import GradScaler as AmpGradScaler
        from torch.amp.autocast_mode import autocast as amp_autocast

        amp_device = "cuda"
        self.scaler = AmpGradScaler(device=amp_device, enabled=self._amp_enabled)
        self._autocast_ctx = lambda: amp_autocast(
            device_type=amp_device,
            dtype=torch.float16,
            enabled=self._amp_enabled,
        )

    def save(self, name: str):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, name)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model: {path}")

    def load(self, name: str):
        path = os.path.join(self.save_dir, name)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Loaded model: {path}")

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.train()
        # 可选：更新 entropy 权重（让 gate 前期不塌缩、后期更“尖”）
        if hasattr(self.criterion, "maybe_update_entropy"):
            try:
                self.criterion.maybe_update_entropy(epoch)
            except Exception:
                pass
        # 可选：用于 path_gate_warmup_epochs（gate_ce 预热）
        if hasattr(self.criterion, "maybe_update_epoch"):
            try:
                self.criterion.maybe_update_epoch(epoch)
            except Exception:
                pass
        pbar = tqdm(self.train_loader, desc=f"Train Ep{epoch+1}")

        sum_total = 0.0
        sum_main = 0.0
        sum_aux = 0.0
        sum_stats: Dict[str, float] = {}

        sum_usage = None
        n_batches = 0
        n_skipped = 0

        for batch in pbar:
            x, y_target, y_abs, init_pos = batch
            x = x.to(self.device, non_blocking=True)
            y_target = y_target.to(self.device, non_blocking=True)
            init_pos = init_pos.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with (self._autocast_ctx() if self._amp_enabled else nullcontext()):
                out = self.model(x, init_pos=init_pos)  # dict
                aux_loss = out.get("aux_loss", 0.0)
                main_loss, loss_dict = self.criterion(out, y_target, x_hist=x)
                if isinstance(aux_loss, torch.Tensor):
                    aux_loss = aux_loss.to(dtype=torch.float32)
                total_loss = (main_loss.to(dtype=torch.float32) + aux_loss).to(dtype=torch.float32)

            if not torch.isfinite(total_loss):
                n_skipped += 1
                continue

            if self._amp_enabled:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            sum_total += float(total_loss.detach().cpu())
            sum_main += float(main_loss.detach().cpu())
            sum_aux += float(aux_loss.detach().cpu()) if isinstance(aux_loss, torch.Tensor) else float(aux_loss)

            for k, v in loss_dict.items():
                try:
                    sum_stats[k] = sum_stats.get(k, 0.0) + float(v)
                except Exception:
                    pass

            usage = out.get("expert_usage", None)
            if usage is not None:
                u = usage.detach().cpu().numpy()
                sum_usage = u if sum_usage is None else (sum_usage + u)

            n_batches += 1
            # 兼容不同 loss 的展示字段（Mixture vs SoftBestK）
            show_nll = loss_dict.get("mix_nll", loss_dict.get("path_nll", loss_dict.get("best_nll", 0.0)))
            show_div = loss_dict.get("div", 0.0)
            show_gate = loss_dict.get("gate_ce", None)
            pbar.set_postfix({
                "Tot": f"{total_loss.item():.3f}",
                "NLL": f"{show_nll:.3f}",
                "Div": f"{show_div:.3f}",
                "Gate": (f"{float(show_gate):.3f}" if show_gate is not None else "-"),
            })

        denom = max(n_batches, 1)

        usage_str = "None"
        if sum_usage is not None:
            usage_str = str(np.round(sum_usage / denom, 3))

        avg_stats = {k: (v / denom) for k, v in sum_stats.items()}
        # 尽量用“最关键”的几个项打印
        if "mix_nll" in avg_stats:
            extra = (
                f"mix_nll={avg_stats.get('mix_nll', 0.0):.4f} "
                f"best_soft={avg_stats.get('best_soft', 0.0):.4f} "
                f"div={avg_stats.get('div', 0.0):.4f} "
                f"entropy={avg_stats.get('entropy', 0.0):.4f}"
            )
        elif "path_nll" in avg_stats:
            extra = (
                f"path_nll={avg_stats.get('path_nll', 0.0):.4f} "
                f"gate_ce={avg_stats.get('gate_ce', 0.0):.4f} "
                f"entropy={avg_stats.get('entropy', 0.0):.4f} "
                f"smooth={avg_stats.get('smooth', 0.0):.4f}"
            )
        else:
            extra = (
                f"best_nll={avg_stats.get('best_nll', 0.0):.4f} "
                f"gate_ce={avg_stats.get('gate_ce', 0.0):.4f} "
                f"div={avg_stats.get('div', 0.0):.4f} "
                f"entropy={avg_stats.get('entropy', 0.0):.4f}"
            )

        logger.info(
            f"[Train] Ep{epoch+1} total={sum_total/denom:.4f} main={sum_main/denom:.4f} aux={sum_aux/denom:.5f}\n"
            f" | {extra}\n"
            f" | expert_usage={usage_str}"
            + (f" | skipped_batches={n_skipped}" if n_skipped > 0 else "")
        )

        out: Dict[str, Any] = {
            "loss": float(sum_total / denom),
            "main": float(sum_main / denom),
            "aux": float(sum_aux / denom),
            "skipped_batches": int(n_skipped),
            **{k: float(v) for k, v in avg_stats.items()},
        }
        if sum_usage is not None:
            out["expert_usage"] = (sum_usage / denom).astype(np.float64).tolist()
        return out

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        sum_total = 0.0
        sum_main = 0.0
        sum_aux = 0.0
        sum_stats: Dict[str, float] = {}

        sum_usage = None
        n_batches = 0

        # 额外：单条“切换轨迹”的 ADE + 平均切换次数
        ade_switch_sum = 0.0
        switch_count_sum = 0.0
        n_samples = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            x, y_target, y_abs, init_pos = batch
            x = x.to(self.device, non_blocking=True)
            y_target = y_target.to(self.device, non_blocking=True)
            y_abs = y_abs.to(self.device, non_blocking=True)
            init_pos = init_pos.to(self.device, non_blocking=True)

            out = self.model(x, init_pos=init_pos)
            aux_loss = out.get("aux_loss", 0.0)
            main_loss, loss_dict = self.criterion(out, y_target, x_hist=x)
            total_loss = main_loss + aux_loss

            sum_total += float(total_loss.detach().cpu())
            sum_main += float(main_loss.detach().cpu())
            sum_aux += float(aux_loss.detach().cpu()) if isinstance(aux_loss, torch.Tensor) else float(aux_loss)

            for k, v in loss_dict.items():
                try:
                    sum_stats[k] = sum_stats.get(k, 0.0) + float(v)
                except Exception:
                    pass

            usage = out.get("expert_usage", None)
            if usage is not None:
                u = usage.detach().cpu().numpy()
                sum_usage = u if sum_usage is None else (sum_usage + u)

            mu_n = out["mu"]                 # (B,K,P,2)
            r_mean = out["r_mean"]           # (B,1,2)
            r_std = out["r_std"]             # (B,1,2)

            mu_m = mu_n * r_std.unsqueeze(1) + r_mean.unsqueeze(1)  # (B,K,P,2) meters/step
            logits_step = out.get("logits_step", None)  # (B,P,K) or None
            if logits_step is None:
                # 退化：没有 per-step gate 时，用全局 gate 的 argmax 作为“整段专家”
                logits_global = out["logits"]                               # (B,K)
                pi_global = torch.softmax(logits_global, dim=-1)            # (B,K)
                k0 = torch.argmax(pi_global, dim=-1)                        # (B,)
                P = int(mu_m.size(2))
                k_steps = k0.unsqueeze(1).expand(-1, P)                     # (B,P)
            else:
                pi_step = torch.softmax(logits_step, dim=-1)                # (B,P,K)
                k_steps = torch.argmax(pi_step, dim=-1)                     # (B,P)

            # mu_m: (B,K,P,2) -> (B,P,2) 逐步 gather
            mu_bpk2 = mu_m.permute(0, 2, 1, 3)                              # (B,P,K,2)
            idx = k_steps.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)  # (B,P,1,2)
            mu_sel = mu_bpk2.gather(2, idx).squeeze(2)                      # (B,P,2)
            pred_abs_sw = torch.cumsum(mu_sel, dim=1) + init_pos.unsqueeze(1)  # (B,P,2)

            err_sw = torch.linalg.norm(pred_abs_sw - y_abs, dim=-1)         # (B,P)
            ade_sw = err_sw.mean(dim=-1)                                    # (B,)
            ade_switch_sum += float(ade_sw.sum().detach().cpu())

            # 切换次数：count(k_t != k_{t-1})
            sw_cnt = (k_steps[:, 1:] != k_steps[:, :-1]).to(torch.float32).sum(dim=-1)  # (B,)
            switch_count_sum += float(sw_cnt.sum().detach().cpu())

            n_samples += x.size(0)
            n_batches += 1

        denom = max(n_batches, 1)
        usage_str = "None"
        if sum_usage is not None:
            usage_str = str(np.round(sum_usage / denom, 3))

        val_loss = sum_total / denom
        # 即便切换次数为 0，也应显示 0.00（而不是 nan）
        switch_ade = ade_switch_sum / max(n_samples, 1)
        switch_cnt = switch_count_sum / max(n_samples, 1)

        avg_stats = {k: (v / denom) for k, v in sum_stats.items()}
        if "mix_nll" in avg_stats:
            extra = (
                f"mix_nll={avg_stats.get('mix_nll', 0.0):.4f} "
                f"best_soft={avg_stats.get('best_soft', 0.0):.4f} "
                f"div={avg_stats.get('div', 0.0):.4f} "
                f"entropy={avg_stats.get('entropy', 0.0):.4f}"
            )
        elif "path_nll" in avg_stats:
            extra = (
                f"path_nll={avg_stats.get('path_nll', 0.0):.4f} "
                f"gate_ce={avg_stats.get('gate_ce', 0.0):.4f} "
                f"entropy={avg_stats.get('entropy', 0.0):.4f} "
                f"smooth={avg_stats.get('smooth', 0.0):.4f}"
            )
        else:
            extra = (
                f"best_nll={avg_stats.get('best_nll', 0.0):.4f} "
                f"gate_ce={avg_stats.get('gate_ce', 0.0):.4f} "
                f"div={avg_stats.get('div', 0.0):.4f} "
                f"entropy={avg_stats.get('entropy', 0.0):.4f}"
            )

        logger.info(
            f"[Val] total={val_loss:.4f} main={sum_main/denom:.4f} aux={sum_aux/denom:.5f} | "
            f"{extra}\n"
            f" | Switch_ADE={switch_ade:.2f}m SwitchCnt={switch_cnt:.2f} | expert_usage={usage_str}"
        )

        # scheduler（通常监控 val_loss 即可；你 Step5 是 ReduceLROnPlateau）
        if self.scheduler is not None:
            self.scheduler.step(val_loss)

        out: Dict[str, Any] = {
            "val_loss": float(val_loss),
            "val_main": float(sum_main / denom),
            "val_aux": float(sum_aux / denom),
            **{k: float(v) for k, v in avg_stats.items()},
            "switch_ade": float(switch_ade),
            "switch_cnt": float(switch_cnt),
        }
        if sum_usage is not None:
            out["expert_usage"] = (sum_usage / denom).astype(np.float64).tolist()
        return out
