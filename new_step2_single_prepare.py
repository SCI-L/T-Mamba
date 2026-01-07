# new_step2_single_prepare.py
# Step2: 载入 Step1 生成的数据 -> 特征工程 -> 划分 Train/Val/Test -> DataLoader

import os
import json
import logging

import numpy as np
import torch
import torch.utils.data as utils

logger = logging.getLogger("Step2")


class SingleShipDataset(utils.Dataset):
    def __init__(self, x, y_target, y_abs, init_pos):
        self.x = torch.FloatTensor(x)
        self.y_target = torch.FloatTensor(y_target)  # (pred, 2) future dx,dy (meters)
        self.y_abs = torch.FloatTensor(y_abs)        # (pred, 2) future absolute (meters)
        self.init_pos = torch.FloatTensor(init_pos)  # (2,) anchor absolute (meters)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_target[idx], self.y_abs[idx], self.init_pos[idx]


def feature_engineering(x_raw, dt=30.0, stats=None):
    """
    [特征工程 - 为 MoE-BiMamba 准备]

    Input: x_raw (N, seq, 4) = [dx, dy, sog(knots), cog(deg)]
    Output: x_feat (N, seq, 11)

    0-1 : dx, dy (Raw meters)
    2   : sog_norm (0~1)
    3-4 : sin(cog), cos(cog)
    5-6 : acc_norm, rot_norm     (AdaRMSNorm 条件)
    7-8 : acc_router_z, rot_router_z  (Router 输入：train-only 标准化后的真实物理量)
    9-10: sog_raw, cog_raw       (Decoder query 初始化)
    """
    dx = x_raw[..., 0:1]
    dy = x_raw[..., 1:2]
    sog = x_raw[..., 2:3]
    cog_deg = x_raw[..., 3:4]

    cog_rad = np.deg2rad(cog_deg)
    sin_c = np.sin(cog_rad)
    cos_c = np.cos(cog_rad)

    # ROT (deg/min)
    diff_cog = cog_rad[:, 1:, :] - cog_rad[:, :-1, :]
    diff_cog = np.mod(diff_cog + np.pi, 2 * np.pi) - np.pi
    rot_steps = diff_cog * (60.0 / float(dt)) * (180.0 / np.pi)

    rot = np.zeros_like(cog_rad)
    rot[:, 1:, :] = rot_steps
    rot[:, 0, :] = rot[:, 1, :]

    # Acc (knots per step)
    acc_steps = sog[:, 1:, :] - sog[:, :-1, :]
    acc = np.zeros_like(sog)
    acc[:, 1:, :] = acc_steps
    acc[:, 0, :] = acc[:, 1, :]

    if stats is None:
        MAX_SOG, MAX_ACC, MAX_ROT = 25.0, 2.0, 40.0
        acc_router, rot_router = acc, rot
    else:
        MAX_SOG = stats["sog_max"]
        MAX_ACC = stats["acc_max"]
        MAX_ROT = stats["rot_max"]

        # Router 的 raw 标准化（train-only mean/std）（z-score）
        acc_router = (acc - stats["acc_raw_mean"]) / (stats["acc_raw_std"] + 1e-6)
        rot_router = (rot - stats["rot_raw_mean"]) / (stats["rot_raw_std"] + 1e-6)
        # 防止少量离群点把 gate 推到极端选择
        acc_router = np.clip(acc_router, -5.0, 5.0)
        rot_router = np.clip(rot_router, -5.0, 5.0)

    sog_norm = np.clip(sog / (float(MAX_SOG) + 1e-6), 0.0, 1.0)
    acc_norm = np.clip(acc / (float(MAX_ACC) + 1e-6), -1.0, 1.0)
    rot_norm = np.clip(rot / (float(MAX_ROT) + 1e-6), -1.0, 1.0)

    x_feat = np.concatenate(
        [
            dx, dy,
            sog_norm,
            sin_c, cos_c,
            acc_norm, rot_norm,
            acc_router, rot_router,
            sog, cog_deg,
        ],
        axis=-1
    ).astype(np.float32)

    return x_feat


def _compute_train_only_stats(x_train_raw, dt=30.0):
    """
    训练集上做 robust statistics（99.9% 分位）来确定 MAX_SOG/ACC/ROT + router 标准化参数
    """
    MIN_SPEED_THRESHOLD = 2.0  # knots

    sog_limit = np.percentile(x_train_raw[..., 2], 99.9)

    temp_sog = x_train_raw[..., 2]
    temp_acc = temp_sog[:, 1:] - temp_sog[:, :-1]
    mask_move_acc = temp_sog[:, 1:] > MIN_SPEED_THRESHOLD
    if np.sum(mask_move_acc) > 0:
        acc_valid = np.abs(temp_acc[mask_move_acc])
        acc_limit = np.percentile(acc_valid, 99.9)
    else:
        acc_limit = 2.0
    acc_limit = min(float(acc_limit), 5.0)

    temp_cog = np.deg2rad(x_train_raw[..., 3])
    temp_diff_cog = temp_cog[:, 1:] - temp_cog[:, :-1]
    temp_diff_cog = np.mod(temp_diff_cog + np.pi, 2 * np.pi) - np.pi
    temp_rot = temp_diff_cog * (60.0 / float(dt)) * (180.0 / np.pi)
    mask_move_rot = temp_sog[:, 1:] > MIN_SPEED_THRESHOLD
    if np.sum(mask_move_rot) > 0:
        rot_valid = np.abs(temp_rot[mask_move_rot])
        rot_limit = np.percentile(rot_valid, 99.9)
    else:
        rot_limit = 30.0
    rot_limit = min(float(rot_limit), 40.0)

    # router mean/std（clip 后）
    if np.sum(mask_move_acc) > 0:
        acc_clip = np.clip(temp_acc[mask_move_acc], -acc_limit, acc_limit)
        acc_mean = float(np.mean(acc_clip))
        acc_std = float(np.std(acc_clip) + 1e-6)
    else:
        acc_mean, acc_std = 0.0, 1.0

    if np.sum(mask_move_rot) > 0:
        rot_clip = np.clip(temp_rot[mask_move_rot], -rot_limit, rot_limit)
        rot_mean = float(np.mean(rot_clip))
        rot_std = float(np.std(rot_clip) + 1e-6)
    else:
        rot_mean, rot_std = 0.0, 1.0

    # dx/dy 的全局均值/方差（meters/step），供 Step3 做归一化（更稳定，也便于 Step4/Step5 反归一化）
    dxdy = x_train_raw[..., 0:2].astype(np.float64)
    dx_mean = float(np.mean(dxdy[..., 0]))
    dy_mean = float(np.mean(dxdy[..., 1]))
    dx_std = float(np.std(dxdy[..., 0]) + 1e-6)
    dy_std = float(np.std(dxdy[..., 1]) + 1e-6)

    return {
        "dxdy_mean": [dx_mean, dy_mean],
        "dxdy_std": [dx_std, dy_std],
        "sog_max": float(sog_limit) + 1e-5,
        "acc_max": float(acc_limit) + 1e-5,
        "rot_max": float(rot_limit) + 1e-5,
        "acc_raw_mean": acc_mean,
        "acc_raw_std": acc_std,
        "rot_raw_mean": rot_mean,
        "rot_raw_std": rot_std,
    }


def get_dataloaders(cfg):
    """
    载入 Step1 的 numpy 数据，并切分 Train/Val/Test：
    - Test：保留一部分 MMSI 完全不参与训练（评估“新船泛化”）
    - Train/Val：剩余 MMSI 再按时间切分（评估“时间泛化”，避免泄露）
    返回：train_dl, val_dl, test_dl, scaler(None), global_stats(dict)
    """
    path_x = os.path.join(cfg.save_dir, "x_train.npy")
    path_id = os.path.join(cfg.save_dir, "ship_ids.npy")

    if not os.path.exists(path_x):
        logger.error(f"缺失数据: {path_x}，请先运行 Step1")
        return None

    if not os.path.exists(path_id):
        logger.error(f"缺失 ID: {path_id}，请先运行 Step1")
        return None

    logger.info(f"加载数据集: {cfg.save_dir}")
    x_raw = np.load(path_x)
    y_target = np.load(os.path.join(cfg.save_dir, "y_offset.npy"))
    y_abs = np.load(os.path.join(cfg.save_dir, "y_abs.npy"))
    init_pos = np.load(os.path.join(cfg.save_dir, "init_pos.npy"))
    ship_ids = np.load(path_id, allow_pickle=True)

    # 载入时间戳（anchor_ts）
    path_anchor_ts = os.path.join(cfg.save_dir, "anchor_ts.npy")
    if not os.path.exists(path_anchor_ts):
        logger.error(f"缺失时间戳: {path_anchor_ts}，请重新运行 Step1（需要输出 anchor_ts.npy）")
        return None
    anchor_ts = np.load(path_anchor_ts)

    unique_ids = np.unique(ship_ids)
    logger.info(f"唯一船舶 MMSI 数: {len(unique_ids)} | 样本数: {len(ship_ids)}")

    # ------------------------------
    # 划分参数（可在 Step5 JSON 中覆盖）
    # ------------------------------
    seed = int(getattr(cfg, "seed", 42))
    test_ship_ratio = float(getattr(cfg, "test_ship_ratio", 0.2))   # Test MMSI 占比（新船）
    val_time_ratio = float(getattr(cfg, "val_time_ratio", 0.15))    # 剩下的船再按时间切分，时间段后 15%做 Val（在非 Test MMSI 内）
    buffer_steps = int(getattr(cfg, "time_split_buffer_steps", cfg.seq_len + cfg.pred_len))

    if not (0.0 < test_ship_ratio < 1.0):
        raise ValueError(f"test_ship_ratio must be in (0,1), got {test_ship_ratio}")
    if not (0.0 < val_time_ratio < 1.0):
        raise ValueError(f"val_time_ratio must be in (0,1), got {val_time_ratio}")

    rng = np.random.default_rng(seed)
    n_test_ships = max(1, int(round(len(unique_ids) * test_ship_ratio)))
    n_test_ships = min(n_test_ships, max(len(unique_ids) - 1, 1))
    test_ship_ids = rng.choice(unique_ids, size=n_test_ships, replace=False)

    is_test_ship = np.isin(ship_ids, test_ship_ids)
    non_test_mask = ~is_test_ship

    # 时间切分：只在非 Test MMSI 内切 Train/Val，且要求窗口完全位于各自时间段（带 buffer）
    dt_val = float(getattr(cfg, "dt", 30.0))
    seq_len = int(getattr(cfg, "seq_len", 40))
    pred_len = int(getattr(cfg, "pred_len", 20))
    window_start_ts = anchor_ts - float(seq_len - 1) * dt_val
    window_end_ts = anchor_ts + float(pred_len) * dt_val

    # 在 cutoff 两侧留一段“隔离区”，隔离区里的样本都不要，从而避免 train 和 val 的窗口在时间上重叠或非常接近。
    # 例如：buffer_sec=600s，意味着在 cutoff 前后各留 600 秒的“隔离区”。
    buffer_sec = float(buffer_steps) * dt_val
    non_test_anchor_ts = anchor_ts[non_test_mask]
    if len(non_test_anchor_ts) == 0:
        logger.error("非 Test MMSI 的样本为空，无法切分 Train/Val。请调小 test_ship_ratio。")
        return None

    # cutoff 取“靠后的分位”，例如 val_time_ratio=0.15 -> cutoff=85% 分位
    # cutoff 是一个时间戳，由非 Test 船样本的 anchor_ts 的分位数决定
    # 从这个时间开始往后算的那段时间，我更希望拿来做验证
    cutoff = float(np.quantile(non_test_anchor_ts, 1.0 - val_time_ratio))

    train_mask = non_test_mask & (window_end_ts <= (cutoff - buffer_sec))
    val_mask = non_test_mask & (window_start_ts >= (cutoff + buffer_sec))
    test_mask = is_test_ship

    # 如果 buffer 太大导致 train/val 为空，自动退化到 buffer=0，并给出 warning
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        logger.warning(
            f"[Split-C] buffer 过大导致集合为空：train={int(train_mask.sum())}, val={int(val_mask.sum())}。"
            f"将临时使用 buffer=0 重新切分（建议调小 time_split_buffer_steps）。"
        )
        train_mask = non_test_mask & (window_end_ts <= cutoff)
        val_mask = non_test_mask & (window_start_ts >= cutoff)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    n_non_test_total = int(non_test_mask.sum())
    n_non_test_kept = int(len(train_idx) + len(val_idx))
    n_non_test_dropped = int(max(n_non_test_total - n_non_test_kept, 0))
    logger.info(
        f"NonTest kept: {n_non_test_kept}/{n_non_test_total} | "
        f"Dropped by cutoff+buffer: {n_non_test_dropped}"
    )

    if len(train_idx) == 0:
        logger.error("训练集为空。请调小 test_ship_ratio/val_time_ratio 或 time_split_buffer_steps。")
        return None
    if len(val_idx) == 0:
        logger.error("验证集为空。请调小 val_time_ratio 或 time_split_buffer_steps（或减少 test_ship_ratio）。")
        return None
    if len(test_idx) == 0:
        logger.warning("测试集为空（test_ship_ratio 太小或船舶数太少）。建议调大 test_ship_ratio。")

    logger.info("计算训练集统计量 (train-only, robust)...")
    train_stats = _compute_train_only_stats(x_raw[train_idx], dt=dt_val)

    # anchor absolute position (ENU meters) train-only stats: for Step3/gate
    # init_pos (from Step1) is the absolute ENU anchor (x,y) per sample.
    try:
        ip = init_pos[train_idx].astype(np.float64)  # (N,2)
        ip_mean = np.mean(ip, axis=0)
        ip_std = np.std(ip, axis=0) + 1e-6
        train_stats["init_pos_mean"] = [float(ip_mean[0]), float(ip_mean[1])]
        train_stats["init_pos_std"] = [float(ip_std[0]), float(ip_std[1])]
    except Exception:
        pass

    logger.info(f"Train stats: {train_stats}")

    stats_path = os.path.join(cfg.save_dir, "train_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(train_stats, f, indent=2, ensure_ascii=False)

    logger.info(f"执行特征工程 (dt={dt_val}s)...")
    x_feat = feature_engineering(x_raw, dt=dt_val, stats=train_stats)
    logger.info(f"x_feat: {x_feat.shape} (期望: N x {cfg.seq_len} x 11)")

    def create_loader(idx, shuffle, batch_size):
        ds = SingleShipDataset(x_feat[idx], y_target[idx], y_abs[idx], init_pos[idx])
        return utils.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=getattr(cfg, "num_workers", 4),
            pin_memory=True,
            persistent_workers=(getattr(cfg, "num_workers", 4) > 0),
        )

    train_bs = cfg.batch_size
    eval_bs = getattr(cfg, "eval_batch_size", min(128, cfg.batch_size))

    train_dl = create_loader(train_idx, True, train_bs)
    val_dl = create_loader(val_idx, False, eval_bs)
    test_dl = create_loader(test_idx, False, eval_bs)

    n_test_unique = len(np.unique(ship_ids[test_idx])) if len(test_idx) > 0 else 0
    n_non_test_unique = len(np.unique(ship_ids[non_test_mask])) if np.any(non_test_mask) else 0
    total_kept = int(len(train_idx) + len(val_idx) + len(test_idx))

    logger.info(f"Total samples (after cutoff+buffer): {total_kept}")
    logger.info(
        f"Test ships: {n_test_unique} | ({len(test_idx)} samples) | \n"
        f"NonTest ships: {n_non_test_unique} | "
        f"Train(time<=cutoff): {len(train_idx)} samples | "
        f"Val(time>=cutoff): {len(val_idx)} samples | "
        f"cutoff_ts={cutoff:.0f}, buffer_steps={buffer_steps}, buffer_sec={buffer_sec:.0f}s"
    )

    scaler_placeholder = None
    return train_dl, val_dl, test_dl, scaler_placeholder, train_stats
