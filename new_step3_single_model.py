# new_step3_single_model.py
# Step3：Bi-Mamba 编码器 + MoE(Experts) 解码器
#
# ========================= 设计说明 =========================
# 目标：对单船历史轨迹进行编码，并输出 K 个未来“专家”(mode/expert) 的分布预测（二维高斯序列）
#
# - 输入：x(B, seq, 11)（由 Step2 生成的特征序列）
# - Encoder：Bi-Mamba（双向 Mamba）堆叠 n_layers，提取“历史运动模式”表征 enc(B,T,D)
# - Decoder：MoEExpertDecoder（expert==mode）
#     1) per-step gate：输出 logits_step(B,P,K)，每个未来步都有一组专家概率 π(t,k)
#     2) experts：输出每个 expert 的未来 P 步二维高斯参数 [mu_x,mu_y,sx,sy,rho]
#
# 关键稳定性设计：
# - dx/dy 位移使用训练集(train-only)统计量做全局归一化（meters/step），避免尺度漂移；
#   并对 dxdy_std 加 clamp_min 防止 std 过小导致数值不稳。
# - AdaRMSNorm ：最后一层 zero-init + tanh 限幅调制（FiLM/AdaNorm 风格），更稳。
# - BiMambaBlock 的 flip 后加 contiguous，避免非连续内存导致的性能/数值问题。
# - gate_feat 使用“全局均值 + 最近窗口均值/方差”，更敏感地捕捉“刚开始转弯/加速”等变化。
# - 输出 dict 的 keys 与 Step4/Step5 兼容。
#
# ========================= 输入特征 x 的索引约定 =========================
# x: (B, seq, 11)
#   0,1   : dx, dy（meters/step）
#   2     : sog_norm（0..1）
#   3,4   : sin_cog, cos_cog
#   5,6   : acc_norm, rot_norm（用于编码器条件归一化）
#   7,8   : acc_router, rot_router（train-only 标准化并 clip；用于 gate）
#   9,10  : sog_raw(knots), cog_deg_raw(deg)（本 Step3 不直接使用）
#
# 输出 dict（供 Step4/Step5 使用）：
#   logits: (B, K)                 全局 gate logits（logits_step 的时间均值；兼容旧接口）
#   logits_step: (B, P, K)         per-step gate logits（用于 switch/load_balance/推理解码）
#   mu:     (B, K, P, 2)           未来每步位移均值（归一化空间）
#   sigma:  (B, K, P, 2)           未来每步位移标准差（归一化空间，>0）
#   rho:    (B, K, P, 1)           相关系数（-1..1）
#   aux_loss: scalar               兼容字段（本实现 encoder 不做 token-level MoE，所以为 0）
#   r_mean: (B, 1, 2)              dx/dy 全局均值（meters/step）
#   r_std:  (B, 1, 2)              dx/dy 全局标准差（meters/step）
#   expert_usage: (K,)             当前 batch 下 π 的平均值（仅用于日志观察“专家是否塌缩/是否均衡”）
# ======================================================================

import math
import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Model")

# --------- feature indices  ---------
IDX_DX = 0
IDX_DY = 1
IDX_SOG_NORM = 2
IDX_SIN_COG = 3
IDX_COS_COG = 4
IDX_ACC_NORM = 5
IDX_ROT_NORM = 6
IDX_ACC_ROUTER = 7
IDX_ROT_ROUTER = 8
IDX_SOG_RAW = 9
IDX_COG_RAW = 10

try:
    from mamba_ssm import Mamba
except Exception as e:
    Mamba = None
    _mamba_import_error = e


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FourierPositionalEncoding(nn.Module):
    pe: torch.Tensor
    """正弦-余弦（sinusoidal）绝对位置编码：给序列加入 sin/cos 位置项，帮助模型区分不同时间步。"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # buffer：不训练，但会跟随 .to(device)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        return x + self.pe[:, : x.size(1), :]


class AdaRMSNorm(nn.Module):
    """
    条件 RMSNorm（FiLM/AdaNorm）：
      1) 对 x 做 RMS 归一化：x_norm = x / rms(x)
      2) 用 cond 生成 (gamma, beta) 对 x_norm 做调制：y = x_norm * (scale + gamma) + beta
      3) 将 cond_proj 的最后一层 zero-init，使初始 gamma/beta≈0 -> 初始行为接近普通 RMSNorm
      4) 对 gamma/beta 使用 tanh 限幅（modulation clamp），防止调制过强导致训练抖动
    """
    def __init__(self, d_model: int, cond_dim: int = 2, eps: float = 1e-6,
                 gamma_max: float = 0.5, beta_max: float = 0.5):
        super().__init__()
        self.eps = eps
        self.gamma_max = float(gamma_max)
        self.beta_max = float(beta_max)

        self.scale = nn.Parameter(torch.ones(d_model))
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        # ---- zero-init 最后一层：初始 gamma/beta≈0，更稳定 ----
        last = self.cond_proj[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D), cond: (B,T,cond_dim) 或 (B,cond_dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms

        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
        # ---- modulation clamp：避免调制过强 ----
        gamma = torch.tanh(gamma) * self.gamma_max
        beta = torch.tanh(beta) * self.beta_max

        return (x_norm * (self.scale + gamma)) + beta


class BiMambaBlock(nn.Module):
    """
    双向 Mamba 块（Bidirectional Mamba）：
      - 前向：mamba_fwd(x)
      - 反向：reverse(mamba_bwd(reverse(x)))
      - 拼接后线性投影回 d_model
    注意：这里的“双向”是指在历史窗口内部同时利用左/右上下文（不是使用未来真实轨迹）。
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        if Mamba is None:
            raise ImportError(f"请先 pip install mamba-ssm\n原始错误: {_mamba_import_error}")
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        y_f = self.fwd(x)
        # flip 后加 contiguous：避免非连续内存带来的性能/潜在数值问题
        x_rev = torch.flip(x, dims=[1]).contiguous()
        y_b = torch.flip(self.bwd(x_rev), dims=[1]).contiguous()
        y = torch.cat([y_f, y_b], dim=-1)
        return self.dropout(self.out_proj(y))


class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN：
        h = silu(Wg x) * (Wv x)
        y = Wo h
    相比 GELU-FFN，通常更强、更稳（尤其在较深网络上）。
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)  # -> split gate/value
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(gate) * val
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class BiMambaLayer(nn.Module):
    """
    编码器单层结构（Pre-Norm + Residual）：
      x = x + LS1 * BiMamba( AdaRMSNorm(x, phys_norm) )
      x = x + LS2 * FFN(     AdaRMSNorm(x, phys_norm) )
    其中 phys_norm 是 Step2 的 [acc_norm, rot_norm]，用于条件归一化。
    LayerScale(LS) 初始很小（1e-3），可显著提升深层训练稳定性。
    """
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 d_ff: int, dropout: float, layerscale_init: float = 1e-3):
        super().__init__()
        self.n1 = AdaRMSNorm(d_model, cond_dim=2)
        self.ssm = BiMambaBlock(d_model, d_state, d_conv, expand, dropout)

        self.n2 = AdaRMSNorm(d_model, cond_dim=2)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

        # LayerScale：每个残差分支一个可学习缩放
        self.ls1 = nn.Parameter(torch.ones(d_model) * layerscale_init)
        self.ls2 = nn.Parameter(torch.ones(d_model) * layerscale_init)

    def forward(self, x: torch.Tensor, phys_norm: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1 * self.ssm(self.n1(x, phys_norm))
        x = x + self.ls2 * self.ffn(self.n2(x, phys_norm))
        return x


class MoEExpertDecoder(nn.Module):
    """
    MoE 解码器：
      - per-step gate：输出 logits_step(B,P,K)，每个未来步一个专家分布 π(t,k)
      - experts：K 个输出头，每个头输出 P 步二维高斯参数 [mu_x, mu_y, sx, sy, rho]
    """
    def __init__(self, d_model: int, pred_len: int, K: int, dropout: float,
                 gate_dim: int, min_sigma_norm: float = 1e-3, layerscale_init: float = 1e-3):
        super().__init__()
        self.pred_len = int(pred_len)
        self.K = int(K)
        self.min_sigma_norm = float(min_sigma_norm)

        # ===== gate（每个未来步一个权重）=====
        self.gate_query = nn.Parameter(torch.zeros(self.pred_len, d_model))  # (P,D)
        nn.init.normal_(self.gate_query, std=0.02)         # P 它不是“真实未来轨迹”，而是 “第 t 个未来步用来查询历史信息的解码 token”
        self.cross_gate = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.gate_norm = nn.LayerNorm(d_model)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model + gate_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, K),
        )
        # 稳定性：gate_mlp 最后一层 zero-init -> 初始 pi_step≈均匀
        last_gate = self.gate_mlp[-1]
        if isinstance(last_gate, nn.Linear):
            nn.init.zeros_(last_gate.weight)
            if last_gate.bias is not None:
                nn.init.zeros_(last_gate.bias)

        # ===== experts =====
        self.query_pos = nn.Parameter(torch.zeros(self.K, self.pred_len, d_model))
        nn.init.normal_(self.query_pos, std=0.02)

        self.cross = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.n2 = nn.LayerNorm(d_model)

        # decoder LayerScale（MoE 下更稳）
        self.ls_attn = nn.Parameter(torch.ones(d_model) * layerscale_init)
        self.ls_ffn = nn.Parameter(torch.ones(d_model) * layerscale_init)

        self.heads = nn.ModuleList([nn.Linear(d_model, 5) for _ in range(self.K)])
        # 稳定性：heads 更保守初始化
        for head in self.heads:
            if isinstance(head, nn.Linear):
                nn.init.zeros_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

    def forward(self, enc: torch.Tensor, gate_feat: torch.Tensor):
        B, T, D = enc.shape
        # (Q,K,V)=(qg, enc, enc),用 cross-attention：qg（Q）去读 enc（K,V）
        qg = self.gate_query.unsqueeze(0).expand(B, -1, -1)  # (B,P,D)
        attn_g, _ = self.cross_gate(qg, enc, enc)
        step_ctx = self.gate_norm(qg + attn_g)               # (B,P,D)，step_ctx（每个未来步一个上下文）
        gate_feat_p = gate_feat.unsqueeze(1).expand(B, self.pred_len, -1)  # (B,P,gate_dim)，gate_feat（整段历史的统计特征）
        
#—————————— 每一步的专家打分 logits_step[b,t,k]，他是由step_ctx和gate_feat_p共同决定的
        logits_step = self.gate_mlp(torch.cat([step_ctx, gate_feat_p], dim=-1))  # (B,P,K)
        logits = logits_step.mean(dim=1)

        q = self.query_pos.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.K, self.pred_len, D)
        enc_rep = enc.unsqueeze(1).expand(B, self.K, T, D).reshape(B * self.K, T, D)

        attn, _ = self.cross(q, enc_rep, enc_rep)
        h = self.n1(q + self.ls_attn * attn)
        h = self.n2(h + self.ls_ffn * self.ffn(h))
        h = h.view(B, self.K, self.pred_len, D)

        raw = torch.stack([self.heads[k](h[:, k, :, :]) for k in range(self.K)], dim=1)  # (B,K,P,5)

        mu = raw[..., 0:2]
        sigma = F.softplus(raw[..., 2:4]) + self.min_sigma_norm
        rho = torch.tanh(raw[..., 4:5]).clamp(-0.999, 0.999)
        return logits, logits_step, mu, sigma, rho


@dataclass
class ModelConfig:
    pred_len: int = 20
    num_modes: int = 6

    d_model: int = 256
    n_layers: int = 6
    dropout: float = 0.1

    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    d_ff: int = 1024

    # dx/dy 全局归一化（meters/step）
    dxdy_mean: Optional[List[float]] = None
    dxdy_std: Optional[List[float]] = None
    dxdy_std_min: float = 1e-3  # clamp_min

    # anchor 绝对位置（ENU meters）的归一化：喂给 gate 用于“在哪儿”这类地理先验
    # 由 Step2 在 train_stats.json 中写入 init_pos_mean/std
    use_init_pos: bool = True
    init_pos_mean: Optional[List[float]] = None
    init_pos_std: Optional[List[float]] = None
    init_pos_std_min: float = 1e-3  # clamp_min

    # gate 最近窗口大小（步数）
    gate_recent_window: int = 4

    aux_loss_coef: float = 0.0


class BiMoEMambaTrajectory(nn.Module):
    dxdy_mean: torch.Tensor
    dxdy_std: torch.Tensor
    init_pos_mean: torch.Tensor
    init_pos_std: torch.Tensor
    pos: FourierPositionalEncoding
    dec: "MoEExpertDecoder"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.dxdy_mean is None or cfg.dxdy_std is None:
            mean = torch.tensor([0.0, 0.0], dtype=torch.float32)
            std = torch.tensor([1.0, 1.0], dtype=torch.float32)
            self._stats_ready = False
        else:
            mean = torch.tensor(cfg.dxdy_mean, dtype=torch.float32)
            std = torch.tensor(cfg.dxdy_std, dtype=torch.float32)
            std = torch.clamp(std, min=float(cfg.dxdy_std_min))  #  clamp_min
            self._stats_ready = True

        self.register_buffer("dxdy_mean", mean.view(1, 1, 2), persistent=True)
        self.register_buffer("dxdy_std", std.view(1, 1, 2), persistent=True)

        # init_pos 归一化（喂给 gate）：如果没有统计量则自动禁用
        if (
            bool(getattr(cfg, "use_init_pos", True))
            and cfg.init_pos_mean is not None
            and cfg.init_pos_std is not None
        ):
            ip_mean = torch.tensor(cfg.init_pos_mean, dtype=torch.float32)
            ip_std = torch.tensor(cfg.init_pos_std, dtype=torch.float32)
            ip_std = torch.clamp(ip_std, min=float(getattr(cfg, "init_pos_std_min", 1e-3)))
            self._init_stats_ready = True
        else:
            ip_mean = torch.tensor([0.0, 0.0], dtype=torch.float32)
            ip_std = torch.tensor([1.0, 1.0], dtype=torch.float32)
            self._init_stats_ready = False

        self.register_buffer("init_pos_mean", ip_mean.view(1, 1, 2), persistent=True)
        self.register_buffer("init_pos_std", ip_std.view(1, 1, 2), persistent=True)

        self.base = nn.Linear(5, cfg.d_model)
        self.pos = FourierPositionalEncoding(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList([
            BiMambaLayer(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # gate_dim = 11：全局均值 + 最近窗口均值/方差
        self._use_init_pos = bool(getattr(cfg, "use_init_pos", True)) and self._init_stats_ready
        self.gate_dim = 11 + (2 if self._use_init_pos else 0)
        self.dec = MoEExpertDecoder(cfg.d_model, cfg.pred_len, cfg.num_modes, cfg.dropout, gate_dim=self.gate_dim)

        logger.info(f"[Step3] BiMambaMoEExpertsTrajectory(K={cfg.num_modes}) params={count_parameters(self)/1e6:.2f}M")

    @staticmethod
    def _safe_std(x: torch.Tensor, dim: int):
        return x.std(dim=dim, unbiased=False)

    def forward(self, x: torch.Tensor, init_pos: Optional[torch.Tensor] = None):
        B, T, _ = x.shape

        if not self._stats_ready and not hasattr(self, "_warned_stats"):
            logger.warning("[Step3] dxdy_mean/std 未设置：当前等价于不做全局归一化。请从训练集统计后传入 ModelConfig.dxdy_mean/std。")
            self._warned_stats = True

        pos_off = x[..., IDX_DX:IDX_DY + 1]
        pos_n = (pos_off - self.dxdy_mean) / (self.dxdy_std + 1e-6)

        others = x[..., IDX_SOG_NORM:IDX_COS_COG + 1]
        base_feat = torch.cat([pos_n, others], dim=-1)
        phys_norm = x[..., IDX_ACC_NORM:IDX_ROT_NORM + 1]

        h = self.drop(self.pos(self.base(base_feat)))
        for layer in self.layers:
            h = layer(h, phys_norm)

        # gate_feat：全局均值 + 最近窗口均值/方差
        speed_mean_all = x[..., IDX_SOG_NORM].mean(dim=1)   # 整段历史平均速度（整体快慢）  
        sin_last = x[:, -1, IDX_SIN_COG]                    # 最后时刻航向（当前朝向）
        cos_last = x[:, -1, IDX_COS_COG]                    # 最后时刻航向（当前朝向）
        acc_mean_all = x[..., IDX_ACC_ROUTER].mean(dim=1)   # 整段历史平均加速度（整体加减速趋势）
        rot_mean_all = x[..., IDX_ROT_ROUTER].mean(dim=1)   # 整段历史平均转向率（整体转弯趋势）

        w = max(1, min(T, int(self.cfg.gate_recent_window)))
        xw = x[:, -w:, :]
        speed_w = xw[..., IDX_SOG_NORM]     
        acc_w = xw[..., IDX_ACC_ROUTER]     
        rot_w = xw[..., IDX_ROT_ROUTER]     

        speed_mean_w = speed_w.mean(dim=1)              # 最近窗口平均速度
        speed_std_w = self._safe_std(speed_w, dim=1)    # 最近窗口速度标准差
        acc_mean_w = acc_w.mean(dim=1)                  # 最近窗口平均加速度
        acc_std_w = self._safe_std(acc_w, dim=1)        # 最近窗口加速度标准差
        rot_mean_w = rot_w.mean(dim=1)                  # 最近窗口平均转向率
        rot_std_w = self._safe_std(rot_w, dim=1)        # 最近窗口转向率标准差

        gate_feat = torch.stack(
            [
                speed_mean_all, sin_last, cos_last, acc_mean_all, rot_mean_all,
                speed_mean_w, speed_std_w, acc_mean_w, acc_std_w, rot_mean_w, rot_std_w,
            ],
            dim=-1,
        )

        # 追加 anchor 绝对位置（ENU meters）给 gate：解决“同样运动学但不同地理位置未来不同”的歧义
        if self._use_init_pos and init_pos is not None:
            ip = init_pos.to(dtype=torch.float32).view(B, 1, 2)
            ip_n = (ip - self.init_pos_mean) / (self.init_pos_std + 1e-6)  # (B,1,2)
            gate_feat = torch.cat([gate_feat, ip_n[:, 0, :]], dim=-1)       # (B,13)

        logits, logits_step, mu, sigma, rho = self.dec(h, gate_feat)

        aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        r_mean = self.dxdy_mean.expand(B, 1, 2)
        r_std = self.dxdy_std.expand(B, 1, 2)
        expert_usage = torch.softmax(logits_step, dim=-1).mean(dim=(0, 1))

        return {
            "logits": logits,
            "logits_step": logits_step,
            "mu": mu,
            "sigma": sigma,
            "rho": rho,
            "aux_loss": aux_loss,
            "r_mean": r_mean,
            "r_std": r_std,
            "expert_usage": expert_usage,
        }


