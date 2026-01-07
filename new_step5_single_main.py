# new_step5_single_main.py
# Step5: 主入口（Step1->Step2->Step3->Step4 训练/验证 + Test 全面评估 + 保存 Step6 的文件）
# 修复：ReduceLROnPlateau(verbose=...) 兼容问题
import os
import json
import time
import random
import logging
import importlib.util
from dataclasses import dataclass, asdict, is_dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.optim as optim

from new_step1_process_single import Step1Config, run_step1_process
from new_step2_single_prepare import get_dataloaders
from new_step3_single_model import ModelConfig, BiMoEMambaTrajectory

# Step4 里应当包含: Trainer + MixtureTrajectoryLoss
from new_step4_single_trainer import (
    Trainer,
    MixtureTrajectoryLoss,
    SoftBestKTrajectoryLoss,
    SingleTrajectoryPathLoss,
)
from moe_decode import viterbi_decode_limited_switch

logger = logging.getLogger("Main")


# ==========================================================
# 1) GlobalConfig
@dataclass
class GlobalConfig:
    data_path: str = "ais_resampled_30s.csv"
    save_dir: str = "./exp_data_single"
    res_dir: str = "./exp_results_single"

    seq_len: int = 40
    pred_len: int = 20
    stride: int = 10
    dt: float = 30.0

    min_traj_points: int = 200
    max_speed_mps: float = 18.0
    max_step_jump_m: float = 600.0
    min_sog_knots: float = 0.3

    # Step1 分段(同一 MMSI 多航次/多天): 用于 Step2 的时间切分与泄露控制
    segment_gap_factor: float = 1.5

    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 4

    # ================================
    # 数据划分(Split-C: 新船 + 时间)
    # ================================
    # Test: 抽取部分 MMSI 完全不参与训练(评估"新船泛化")
    test_ship_ratio: float = 0.2
    # Val: 在非 Test MMSI 内按时间切分(评估"时间泛化")
    val_time_ratio: float = 0.15
    # 时间切分安全缓冲(步): 避免 train/val 窗口跨越 cutoff(默认一个窗口长度)
    time_split_buffer_steps: int = 60

    # Model
    d_model: int = 256
    n_layers: int = 6
    dropout: float = 0.1
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    d_ff: int = 1024
    num_experts: int = 8
    top_k: int = 2
    aux_loss_coef: float = 0.01
    gate_recent_window: int = 4
    use_init_pos: bool = True
    dxdy_std_min: float = 1e-3
    init_pos_std_min: float = 1e-3

    # 专家数(K)
    num_modes: int = 6

    # ================================
    # 推理解码(per-step gate -> 单条轨迹)
    # ================================
    # max_switches: 最多允许切换次数(段数=切换+1); 越小越平稳
    viterbi_max_switches: int = 2
    # switch_cost：切换惩罚（越大越不愿切换）
    viterbi_switch_cost: float = 1.0
    # （不再保存“多条候选轨迹”的文件；只输出单条解码轨迹 + 热力图）

    # Step6
    heatmap_grid: int = 260

    # Train
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-2
    use_amp: bool = True
    grad_clip: float = 1.0
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    plot_every_epochs: int = 10
    seed: int = 42

    loss_cfg: Dict[str, Any] = None
    # loss_type:
    # - "path":   单条轨迹训练(Viterbi 路径对齐: 专家只学自己那条 + gate 对齐, 最贴近部署的"分段切换专家")
    # - "mixture": Mixture NLL(log-sum-exp 的加权分布训练, 更"软", 但与单条轨迹解码存在目标差异)
    # - "softbestk": Soft-best-of-K(更偏"选一个最好的专家"训练方式, 部署时可配合 top-1/top-2)
    loss_type: str = "path"


def default_loss_cfg():
    # 推荐: per-step Mixture SOTA 配置(你可按需调)
    return {
        # Mixture NLL
        "nll": 1.0,
        # Soft best-of-K(用 MSE 做辅助回归, 防止所有 mode 都学不好)
        "soft_bestofk": 0.15,
        # 多样性(mode 终点分散, 防塌缩)
        "diversity": 0.02,
        # entropy(建议小且衰减到 0: 前期防塌缩, 后期让 gate 变尖)
        "entropy": 0.001,
        "entropy_decay_epochs": 10,
        # per-step gate: 抑制频繁切换 + load balance
        "switch": 0.02,
        "load_balance": 0.01,
        "path_load_balance": 0.01,
        "path_smooth_lam": 0.0,

        "softmin_tau": 0.5,
        "min_sigma_m": 0.8,

        # SoftBestK 兼容字段(如 loss_type=softbestk 会用到)
        "best_nll": 1.0,
        "gate_ce": 0.1,
        "tau_gate": 1.0,
        "path_entropy": 0.001,
        "path_entropy_decay_epochs": 0,
    }


def _load_cfg_dict_from_py(py_path: str) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("user_config", py_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Invalid python config: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "get_config"):
        d = module.get_config()
    elif hasattr(module, "CONFIG"):
        d = module.CONFIG
    else:
        raise ValueError("Python config must define CONFIG dict or get_config().")

    if is_dataclass(d):
        d = asdict(d)
    if not isinstance(d, dict):
        raise ValueError("Python config must return a dict.")
    return d


def load_cfg(config_path: str) -> GlobalConfig:
    cfg = GlobalConfig()
    cfg.loss_cfg = default_loss_cfg()

    if config_path is None:
        return cfg

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()
    if ext == ".py":
        d = _load_cfg_dict_from_py(config_path)
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            d = json.load(f)

    for k, v in d.items():
        if k == "loss_cfg" and isinstance(v, dict):
            cfg.loss_cfg.update(v)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)

    if cfg.loss_cfg is None:
        cfg.loss_cfg = default_loss_cfg()
    return cfg


# ==========================================================
# 2) 工具函数
# ==========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ==========================================================
# 3) Step1 智能检测
# ==========================================================
def check_and_prepare_data(cfg: GlobalConfig) -> None:
    _ensure_dir(cfg.save_dir)
    config_path = os.path.join(cfg.save_dir, "data_config.json")

    required = ["x_train.npy", "y_offset.npy", "y_abs.npy", "init_pos.npy", "ship_ids.npy", "segment_ids.npy", "anchor_ts.npy"]
    data_exists = all(os.path.exists(os.path.join(cfg.save_dir, f)) for f in required)

    should_run = False
    reason = ""

    if (not data_exists) or (not os.path.exists(config_path)):
        should_run = True
        reason = "数据文件缺失"
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                old_cfg = json.load(f)
            if old_cfg.get("seq_len") != cfg.seq_len or \
               old_cfg.get("pred_len") != cfg.pred_len or \
               old_cfg.get("stride") != cfg.stride or \
               old_cfg.get("dt") != cfg.dt or \
              old_cfg.get("segment_gap_factor") != cfg.segment_gap_factor:
                should_run = True
                reason = "关键参数变更 (seq/pred/stride/dt/segment)"
            else:
                # 坐标系/投影方法变更也需要重跑 Step1(避免新代码读取旧坐标数据)
                expected_method = "enu_wgs84"
                geo = old_cfg.get("geo_to_xy", None)
                old_method = geo.get("method", None) if isinstance(geo, dict) else None
                if old_method != expected_method:
                    should_run = True
                    reason = f"坐标投影变更 (geo_to_xy.method: {old_method} -> {expected_method})"
                else:
                    logger.info("[Step5] 检测到匹配的 Step1 数据, 跳过 Step1")
        except Exception as e:
            should_run = True
            reason = f"旧配置读取失败: {e}"

    if should_run:
        logger.info(f"[Step5] 将重新运行 Step1(原因: {reason})")
        s1 = Step1Config(
            data_path=cfg.data_path,
            save_dir=cfg.save_dir,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            stride=cfg.stride,
            dt=cfg.dt,
            min_traj_points=cfg.min_traj_points,
            max_speed_mps=cfg.max_speed_mps,
            max_step_jump_m=cfg.max_step_jump_m,
            min_sog_knots=cfg.min_sog_knots,
            segment_gap_factor=cfg.segment_gap_factor,
        )
        run_step1_process(s1)


# ==========================================================
# 4) 评估指标（单条轨迹：ADE/FDE + 运动学误差）
# ==========================================================
def calc_kinematics_numpy(traj: np.ndarray, dt: float = 30.0):
    """
    traj: (N, T, 2) 绝对坐标(米)
    return: sog(knots) (N,T-1), cog(deg) (N,T-1)
    """
    diffs = traj[:, 1:] - traj[:, :-1]
    dists = np.linalg.norm(diffs, axis=-1)
    sogs = (dists / dt) * 1.9438444924406
    cogs = np.degrees(np.arctan2(diffs[..., 1], diffs[..., 0]))
    cogs = (cogs + 360.0) % 360.0
    return sogs, cogs


def calc_cog_mae(pred_cog: np.ndarray, true_cog: np.ndarray) -> float:
    diff = np.abs(pred_cog - true_cog)
    diff = np.minimum(diff, 360.0 - diff)
    return float(np.mean(diff))


def build_hist_abs_from_offsets(anchor: np.ndarray, hist_off: np.ndarray) -> np.ndarray:
    """
    anchor: (B,2) 预测起点绝对坐标(最后一个历史点)
    hist_off: (B,seq,2) 历史每步位移(meters/step)
    return hist_abs: (B,seq+1,2)
    """
    B, seq, _ = hist_off.shape
    out = np.zeros((B, seq + 1, 2), dtype=np.float64)
    for i in range(B):
        curr = anchor[i].copy()
        path = [curr.copy()]
        for off in reversed(hist_off[i]):
            curr = curr - off
            path.append(curr.copy())
        out[i] = np.array(path[::-1], dtype=np.float64)
    return out


@torch.no_grad()
def run_full_evaluation(model, loader, device: str, cfg: GlobalConfig) -> Dict[str, float]:
    """
    Step5 Test 评估(支持 per-step gate -> 单条"切换轨迹")
    - 主输出: 通过 Viterbi(受限切换) 从 pi_step(t,k) 解码得到 k_t, 再拼接成单条轨迹
      保存到 Step6(test_pred.npy/test_dist_params.npy/test_hist.npy/test_true.npy)
    - 可选: 额外保存 K 条 mode 轨迹与其分布参数(用于分析/调试)
    """
    model.eval()
    logger.info("[Step5] 开始全面评估 (Test, 单条轨迹解码) ...")

    pred_single_list = []
    targets = []
    hists = []

    # Step6 需要的文件（单条轨迹）
    single_params_list = []  # (N,P,5) [x,y,sx,sy,rho]
    single_kin_list = []     # (N,P,2) [SOG(knots), ROT(deg/min)]

    pi_steps_list = []       # (N,P,K)
    k_steps_list = []        # (N,P)
    switch_cnt_list = []     # (N,)

    # 不再保存"多条候选轨迹上界指标"; 只保留单条解码轨迹与 gate 解释性输出
    for batch in loader:
        x, y_target, y_abs, init_pos = [b.to(device) for b in batch]

        out = model(x, init_pos=init_pos)
        logits_global = out["logits"]    # (B,K)
        logits_step = out.get("logits_step", None)  # (B,P,K) or None
        mu_n = out["mu"]                 # (B,K,P,2) norm space
        sigma_n = out["sigma"]           # (B,K,P,2) norm space
        rho = out["rho"]                 # (B,K,P,1)
        r_mean = out["r_mean"]           # (B,1,2)
        r_std = out["r_std"]             # (B,1,2)

        # per-step gate 概率
        if logits_step is None:
            pi_global_t = torch.softmax(logits_global, dim=-1)  # (B,K)
            # 退化: 复制成每一步相同的 gate(便于统一解码)
            pi_step = pi_global_t.unsqueeze(1).repeat(1, y_abs.size(1), 1)  # (B,P,K)
        else:
            pi_step = torch.softmax(logits_step, dim=-1)  # (B,P,K)

        # 反归一化到 meters/step
        mu_m = mu_n * r_std.unsqueeze(1) + r_mean.unsqueeze(1)    # (B,K,P,2)
        sigma_m = sigma_n * r_std.unsqueeze(1)                    # (B,K,P,2)

        mu_m_np = mu_m.detach().cpu().numpy().astype(np.float64)
        sigma_m_np = sigma_m.detach().cpu().numpy().astype(np.float64)
        rho_np = rho.detach().cpu().numpy().astype(np.float64)
        pi_step_np = pi_step.detach().cpu().numpy().astype(np.float64)  # (B,P,K)

        anchor = init_pos.detach().cpu().numpy().astype(np.float64)     # (B,2)
        true_abs = y_abs.detach().cpu().numpy().astype(np.float64)      # (B,P,2)

        # ========== 受限切换 Viterbi: 每条样本得到 k_steps(P,) ==========
        B, _, P, _ = mu_m_np.shape
        K = int(mu_m_np.shape[1])

        k_steps = np.zeros((B, P), dtype=np.int64)
        sw_cnt = np.zeros((B,), dtype=np.int64)
        for i in range(B):
            res = viterbi_decode_limited_switch(
                pi_step_np[i],
                max_switches=int(getattr(cfg, "viterbi_max_switches", 2)),
                switch_cost=float(getattr(cfg, "viterbi_switch_cost", 1.0)),
            )
            k_steps[i] = res.k_steps
            sw_cnt[i] = int(res.n_switches)

        # 解释性: 保存每步 gate 与解码后的专家序列
        pi_steps_list.append(pi_step_np)
        k_steps_list.append(k_steps)
        switch_cnt_list.append(sw_cnt)

        # ========== 单条"切换轨迹"拼接 ==========
        # mu_sel/sigma_sel/rho_sel: (B,P,*) 逐步根据 k_steps 选择
        mu_bpk2 = np.transpose(mu_m_np, (0, 2, 1, 3))       # (B,P,K,2)
        sigma_bpk2 = np.transpose(sigma_m_np, (0, 2, 1, 3))  # (B,P,K,2)
        rho_bpk1 = np.transpose(rho_np, (0, 2, 1, 3))        # (B,P,K,1)

        idx = k_steps[:, :, None, None]  # (B,P,1,1)
        idx2 = np.repeat(idx, 2, axis=3)  # (B,P,1,2)

        mu_sel = np.take_along_axis(mu_bpk2, idx2, axis=2)[:, :, 0, :]          # (B,P,2)
        sigma_sel = np.take_along_axis(sigma_bpk2, idx2, axis=2)[:, :, 0, :]    # (B,P,2)
        rho_sel = np.take_along_axis(rho_bpk1, idx, axis=2)[:, :, 0, 0]         # (B,P)

        pred_abs_single = np.cumsum(mu_sel, axis=1) + anchor[:, None, :]        # (B,P,2)
        pred_single_list.append(pred_abs_single)
        targets.append(true_abs)

        # 单条轨迹的位置分布：累计协方差（独立步近似）
        sx = sigma_sel[..., 0]
        sy = sigma_sel[..., 1]
        varx_cum = np.cumsum(sx * sx, axis=1)
        vary_cum = np.cumsum(sy * sy, axis=1)
        covxy_cum = np.cumsum(rho_sel * sx * sy, axis=1)
        sx_cum = np.sqrt(varx_cum)
        sy_cum = np.sqrt(vary_cum)
        rho_cum = covxy_cum / (sx_cum * sy_cum + 1e-12)
        rho_cum = np.clip(rho_cum, -0.999, 0.999)

        params_single = np.concatenate(
            [
                pred_abs_single,
                sx_cum[..., None],
                sy_cum[..., None],
                rho_cum[..., None],
            ],
            axis=-1,
        )  # (B,P,5)
        single_params_list.append(params_single)

        # ========== 历史绝对轨迹(给 Step6) ==========
        x_np = x.detach().cpu().numpy().astype(np.float64)
        hist_off = x_np[:, :, 0:2]  # meters/step
        hist_abs = build_hist_abs_from_offsets(anchor, hist_off)
        hists.append(hist_abs)

        # 动力学（单条轨迹）：SOG/ROT
        dists = np.linalg.norm(mu_sel, axis=-1)
        pred_sog = (dists / cfg.dt) * 1.9438444924406  # (B,P)
        pred_cog = np.degrees(np.arctan2(mu_sel[..., 1], mu_sel[..., 0]))
        pred_cog = (pred_cog + 360.0) % 360.0
        rot_deg_step = np.zeros_like(pred_cog)
        rot_deg_step[:, 1:] = pred_cog[:, 1:] - pred_cog[:, :-1]
        rot_deg_step = (rot_deg_step + 180.0) % 360.0 - 180.0
        rot_deg_min = rot_deg_step * (60.0 / cfg.dt)
        single_kin_list.append(np.stack([pred_sog, rot_deg_min], axis=-1))

    # concat
    P_single = np.concatenate(pred_single_list, axis=0)
    T = np.concatenate(targets, axis=0)
    H = np.concatenate(hists, axis=0)
    PARAM_single = np.concatenate(single_params_list, axis=0)
    KIN_single = np.concatenate(single_kin_list, axis=0)

    PI_steps = np.concatenate(pi_steps_list, axis=0)   # (N,P,K)
    K_steps = np.concatenate(k_steps_list, axis=0)     # (N,P)
    SW_cnt = np.concatenate(switch_cnt_list, axis=0)   # (N,)

    # pi_global(用于快速条形图): 时间均值
    PI_global = PI_steps.mean(axis=1)                  # (N,K)
    top1_mode_all = np.argmax(PI_global, axis=1).astype(np.int64)             # (N,)
    top1_pi_all = PI_global[np.arange(len(PI_global)), top1_mode_all]         # (N,)
    pi_entropy_all = -(PI_global * np.log(PI_global + 1e-12)).sum(axis=1)     # (N,)
    pi_step_entropy = -(PI_steps * np.log(PI_steps + 1e-12)).sum(axis=2).mean(axis=1)  # (N,)

    # 单条轨迹 ADE/FDE
    err_single = np.linalg.norm(P_single - T, axis=-1)
    ADE = float(err_single.mean())
    FDE = float(err_single[:, -1].mean())

    # 动力学(单条轨迹)
    anchors = H[:, -1:, :]
    P_full = np.concatenate([anchors, P_single], axis=1)
    T_full = np.concatenate([anchors, T], axis=1)
    p_sog, p_cog = calc_kinematics_numpy(P_full, cfg.dt)
    t_sog, t_cog = calc_kinematics_numpy(T_full, cfg.dt)
    sog_rmse = float(np.sqrt(np.mean((p_sog - t_sog) ** 2)))
    cog_mae = calc_cog_mae(p_cog, t_cog)

    metrics = {
        "ADE": ADE,
        "FDE": FDE,
        "SOG_RMSE": sog_rmse,
        "COG_MAE": cog_mae,
        "Avg_Top1_pi": float(np.mean(top1_pi_all)),
        "Avg_pi_entropy": float(np.mean(pi_entropy_all)),
        "Avg_pi_step_entropy": float(np.mean(pi_step_entropy)),
        "Avg_switch_count": float(np.mean(SW_cnt)),
    }

    logger.info("=" * 36)
    logger.info("最终测试指标 (Test Metrics, Single Trajectory):")
    logger.info(f"  ADE: {metrics['ADE']:.2f} m")
    logger.info(f"  FDE: {metrics['FDE']:.2f} m")
    logger.info(f"  SOG_RMSE: {metrics['SOG_RMSE']:.2f} knots")
    logger.info(f"  COG_MAE:  {metrics['COG_MAE']:.2f} deg")
    logger.info(f"  Avg_Top1_pi(mean over t): {metrics['Avg_Top1_pi']:.3f}")
    logger.info(f"  Avg_pi_entropy(global):   {metrics['Avg_pi_entropy']:.3f}")
    logger.info(f"  Avg_pi_step_entropy:      {metrics['Avg_pi_step_entropy']:.3f}")
    logger.info(f"  Avg_switch_count:         {metrics['Avg_switch_count']:.2f}")
    logger.info("=" * 36)

    # 保存: Step6 主输出(单条轨迹)
    _ensure_dir(cfg.res_dir)
    np.save(os.path.join(cfg.res_dir, "test_pred.npy"), P_single)
    np.save(os.path.join(cfg.res_dir, "test_true.npy"), T)
    np.save(os.path.join(cfg.res_dir, "test_hist.npy"), H)
    np.save(os.path.join(cfg.res_dir, "test_dist_params.npy"), PARAM_single)
    np.save(os.path.join(cfg.res_dir, "test_kinematics.npy"), KIN_single)

    # gate 解释性输出(per-step)
    np.save(os.path.join(cfg.res_dir, "test_pi_steps.npy"), PI_steps)         # (N,P,K)
    np.save(os.path.join(cfg.res_dir, "test_k_steps.npy"), K_steps)           # (N,P)
    np.save(os.path.join(cfg.res_dir, "test_switch_count.npy"), SW_cnt)       # (N,)

    # 兼容: 给 Step6 用的 pi 条形图(全局=时间均值)
    np.save(os.path.join(cfg.res_dir, "test_pi.npy"), PI_global)              # (N,K)
    np.save(os.path.join(cfg.res_dir, "test_top1_mode.npy"), top1_mode_all)   # (N,)
    np.save(os.path.join(cfg.res_dir, "test_top1_pi.npy"), top1_pi_all)       # (N,)
    np.save(os.path.join(cfg.res_dir, "test_pi_entropy.npy"), pi_entropy_all) # (N,)

    with open(os.path.join(cfg.res_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


# ==========================================================
# 5) 主流程
# ==========================================================
class TrainingHistory:
    def __init__(self):
        self.history: Dict[str, list] = {}

    def _append(self, key: str, value: float):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(float(value))

    def update(self, train_out: Dict[str, Any], val_out: Dict[str, Any]):
        # train
        if "loss" in train_out:
            self._append("train_loss", float(train_out["loss"]))
        if "main" in train_out:
            self._append("train_main", float(train_out["main"]))
        if "aux" in train_out:
            self._append("train_aux", float(train_out["aux"]))
        if "skipped_batches" in train_out:
            self._append("train_skipped_batches", float(train_out["skipped_batches"]))
        for k in ["path_nll", "gate_ce", "path_switch", "entropy", "lb", "mix_nll", "best_soft", "div"]:
            if k in train_out:
                self._append(f"train_{k}", float(train_out[k]))

        # val
        if "val_loss" in val_out:
            self._append("val_loss", float(val_out["val_loss"]))
        if "val_main" in val_out:
            self._append("val_main", float(val_out["val_main"]))
        if "val_aux" in val_out:
            self._append("val_aux", float(val_out["val_aux"]))
        for k in ["path_nll", "gate_ce", "path_switch", "entropy", "lb", "mix_nll", "best_soft", "div", "switch_ade", "switch_cnt"]:
            if k in val_out:
                self._append(f"val_{k}", float(val_out[k]))

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def plot(self, save_path: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_loss = self.history.get("train_loss", [])
        val_loss = self.history.get("val_loss", [])
        if len(train_loss) == 0:
            return

        epochs = list(range(1, len(train_loss) + 1))
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(epochs, train_loss, label="Train loss", linewidth=2.0)
        if len(val_loss) == len(train_loss) and len(val_loss) > 0:
            plt.plot(epochs, val_loss, label="Val loss", linewidth=2.0)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_extra(self, save_path: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_loss = self.history.get("train_loss", [])
        if len(train_loss) == 0:
            return
        epochs = list(range(1, len(train_loss) + 1))

        panels = [
            ("Loss", ("train_loss", "val_loss")),
            ("path_nll", ("train_path_nll", "val_path_nll")),
            ("gate_ce", ("train_gate_ce", "val_gate_ce")),
            ("path_switch", ("train_path_switch", "val_path_switch")),
            ("entropy", ("train_entropy", "val_entropy")),
            ("lb", ("train_lb", "val_lb")),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=120)
        axes = axes.reshape(-1)
        for ax, (title, (k_tr, k_va)) in zip(axes, panels):
            tr = self.history.get(k_tr, [])
            va = self.history.get(k_va, [])
            if len(tr) == len(epochs) and len(tr) > 0:
                ax.plot(epochs, tr, label="train", linewidth=2.0)
            if len(va) == len(epochs) and len(va) > 0:
                ax.plot(epochs, va, label="val", linewidth=2.0)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.25)
            if (len(tr) > 0) or (len(va) > 0):
                ax.legend()
        plt.tight_layout()
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def main(config_path: str = None):
    cfg = load_cfg(config_path)

    _ensure_dir(cfg.res_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(cfg.res_dir, "run.log"), mode="w", encoding="utf-8"),
        ],
        force=True,
    )

    logger.info("=" * 60)
    logger.info("[Step5] Bi-MoE-Mamba 单条轨迹预测流程")
    logger.info(f"config_path: {config_path}")
    logger.info(f"save_dir   : {cfg.save_dir}")
    logger.info(f"res_dir    : {cfg.res_dir}")
    logger.info("=" * 60)

    # 配置快照
    with open(os.path.join(cfg.res_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Step1
    check_and_prepare_data(cfg)

    # Step2
    out = get_dataloaders(cfg)
    if out is None:
        raise RuntimeError("Step2 数据加载失败, 请检查 Step1 输出")
    train_dl, val_dl, test_dl, _, train_stats = out

    # Step3
    # Step3 当前实现里: expert = mode(K=num_modes), 不再需要 num_experts/top_k 这类"路由"参数;
    # 这些字段仍保留在 JSON 里只是为了配置兼容可读性.
    dxdy_mean = None
    dxdy_std = None
    init_pos_mean = None
    init_pos_std = None
    if isinstance(train_stats, dict):
        dxdy_mean = train_stats.get("dxdy_mean", None)
        dxdy_std = train_stats.get("dxdy_std", None)
        init_pos_mean = train_stats.get("init_pos_mean", None)
        init_pos_std = train_stats.get("init_pos_std", None)

    mcfg = ModelConfig(
        pred_len=cfg.pred_len,
        num_modes=cfg.num_modes,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        d_ff=cfg.d_ff,
        aux_loss_coef=cfg.aux_loss_coef,
        dxdy_mean=dxdy_mean,
        dxdy_std=dxdy_std,
        dxdy_std_min=cfg.dxdy_std_min,
        use_init_pos=cfg.use_init_pos,
        init_pos_mean=init_pos_mean,
        init_pos_std=init_pos_std,
        init_pos_std_min=cfg.init_pos_std_min,
        gate_recent_window=cfg.gate_recent_window,
    )
    model = BiMoEMambaTrajectory(mcfg).to(device)

    # Step4 loss/optim
    loss_type = str(getattr(cfg, "loss_type", "path")).lower()
    if loss_type in ["mixture", "mix", "mdn"]:
        criterion = MixtureTrajectoryLoss(cfg.loss_cfg)
        logger.info("[Step4] Loss: MixtureTrajectoryLoss (mixture NLL)")
    elif loss_type in ["softbestk", "bestk", "soft_bestk", "soft-bestk"]:
        criterion = SoftBestKTrajectoryLoss(cfg.loss_cfg)
        logger.info("[Step4] Loss: SoftBestKTrajectoryLoss (soft-best-of-K NLL + gate_ce)")
    elif loss_type in ["path", "expert_path", "viterbi", "single_path", "single-trajectory"]:
        criterion = SingleTrajectoryPathLoss(cfg.loss_cfg)
        logger.info("[Step4] Loss: SingleTrajectoryPathLoss (Viterbi path-aligned single trajectory)")
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Use 'path' (recommended) / 'mixture' / 'softbestk'."
        )
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 修复: 不使用 verbose(你环境的 torch 不支持)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        save_dir=cfg.res_dir,
        use_amp=cfg.use_amp,
        grad_clip=cfg.grad_clip,
    )

    history = TrainingHistory()
    best_val = 1e9
    best_epoch = -1

    logger.info(f"开始训练 {cfg.epochs} Epochs...")
    t0 = time.time()

    for epoch in range(cfg.epochs):
        train_out = trainer.train_epoch(epoch)
        val_out = trainer.validate()

        if not isinstance(train_out, dict):
            train_out = {"loss": float(train_out)}
        if not isinstance(val_out, dict):
            val_out = {"val_loss": float(val_out)}

        val_loss = float(val_out.get("val_loss", val_out.get("loss", 0.0)))

        history.update(train_out, val_out)
        history.save(os.path.join(cfg.res_dir, "training_history.json"))

        # 提速：训练曲线不必每个 epoch 都画（matplotlib + IO 会明显拖慢）
        plot_every = max(int(cfg.plot_every_epochs), 0)
        should_plot = (
            plot_every > 0
            and ((epoch + 1) % plot_every == 0 or epoch == 0 or (epoch + 1) == int(cfg.epochs))
        )
        if should_plot:
            history.plot(os.path.join(cfg.res_dir, "training_curve.png"))
            history.plot_extra(os.path.join(cfg.res_dir, "training_curves_extra.png"))

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            trainer.save("best_model.pt")

        train_loss = float(train_out.get("loss", 0.0))
        logger.info(f"[Epoch {epoch+1:03d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} (best={best_val:.4f}@{best_epoch+1})")

    logger.info(f"训练完成，总耗时: {(time.time()-t0)/60.0:.1f} min")
    logger.info(f"Best Val Loss: {best_val:.4f} @ Epoch {best_epoch+1}")

    # Ensure final curves are saved even when plot_every_epochs=0.
    try:
        history.plot(os.path.join(cfg.res_dir, "training_curve.png"))
        history.plot_extra(os.path.join(cfg.res_dir, "training_curves_extra.png"))
    except Exception:
        pass

    # Test
    trainer.load("best_model.pt")
    metrics = run_full_evaluation(model, test_dl, device, cfg)
    logger.info(f"[Step5] Done 结果已保存到: {cfg.res_dir}")
    return metrics


if __name__ == "__main__":
    import argparse

    # 便捷: 如果用户未显式传 --config, 但目录下存在默认配置文件, 则自动使用它
    default_cfg = None
    for cand in ["new_config_bi_moe_mamba_single.py", "new_config_bi_moe_mamba_single.json"]:
        if os.path.exists(cand):
            default_cfg = cand
            break

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_cfg, help="path to config json/py")
    args = parser.parse_args()
    main(args.config)
