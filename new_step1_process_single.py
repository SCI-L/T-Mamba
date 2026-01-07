# new_step1_process_single.py
# Step1: 读取 AIS 原始数据 -> 轨迹清洗 -> 滑窗切片 -> 保存训练所需的 numpy 文件
# 目标：单船（每条样本仅包含一艘船的历史与未来）。

import os
import json
import math
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("Step1")


# ==========================================================
# 0) 经纬度 <-> 局部米坐标（WGS84 ECEF->ENU，本地东-北坐标，单位：米）
# ==========================================================
WGS84_A = 6378137.0  # semi-major axis (m)
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: float = 0.0):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + h_m) * cos_lat * cos_lon
    y = (N + h_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + h_m) * sin_lat
    return x, y, z


def _ecef_to_geodetic(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    ECEF -> geodetic lat/lon (deg) + height (m).
    Iterative method; stable for local visualization.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    lon = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    h = np.zeros_like(lat)

    for _ in range(6):
        sin_lat = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / (np.cos(lat) + 1e-15) - N
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * (N / (N + h + 1e-15))))

    return np.rad2deg(lat), np.rad2deg(lon), h

def latlon_to_local_xy_m(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    将经纬度（度）转换到局部平面米坐标（x East, y North）。
    适用于研究区域相对不大的场景（AIS 常见）。
    """
    x_ecef, y_ecef, z_ecef = _geodetic_to_ecef(lat, lon, h_m=0.0)
    x0, y0, z0 = _geodetic_to_ecef(np.array([lat0]), np.array([lon0]), h_m=0.0)
    x0, y0, z0 = float(x0[0]), float(y0[0]), float(z0[0])

    dx = x_ecef - x0
    dy = y_ecef - y0
    dz = z_ecef - z0

    lat0_rad = math.radians(float(lat0))
    lon0_rad = math.radians(float(lon0))
    slat, clat = math.sin(lat0_rad), math.cos(lat0_rad)
    slon, clon = math.sin(lon0_rad), math.cos(lon0_rad)

    east = -slon * dx + clon * dy
    north = -slat * clon * dx - slat * slon * dy + clat * dz
    return east.astype(np.float32), north.astype(np.float32)

def local_xy_to_latlon(x: np.ndarray, y: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """局部米坐标 -> 经纬度（度）。用于可视化阶段还原。"""
    e = np.asarray(x, dtype=np.float64)
    n = np.asarray(y, dtype=np.float64)
    u = np.zeros_like(e)

    lat0_rad = math.radians(float(lat0))
    lon0_rad = math.radians(float(lon0))
    slat, clat = math.sin(lat0_rad), math.cos(lat0_rad)
    slon, clon = math.sin(lon0_rad), math.cos(lon0_rad)

    dx = -slon * e - slat * clon * n + clat * clon * u
    dy = clon * e - slat * slon * n + clat * slon * u
    dz = clat * n + slat * u

    x0, y0, z0 = _geodetic_to_ecef(np.array([lat0]), np.array([lon0]), h_m=0.0)
    x_ecef = dx + float(x0[0])
    y_ecef = dy + float(y0[0])
    z_ecef = dz + float(z0[0])

    lat_deg, lon_deg, _h = _ecef_to_geodetic(x_ecef, y_ecef, z_ecef)
    return lat_deg.astype(np.float64), lon_deg.astype(np.float64)


# ==========================================================
# 1) Step1 配置
# ==========================================================
@dataclass
class Step1Config:
    # 数据路径：支持 CSV / Parquet / Pickle(DataFrame)
    data_path: str
    save_dir: str

    # 序列长度
    seq_len: int = 40
    pred_len: int = 20
    stride: int = 10
    dt: float = 30.0  # seconds

    # 清洗阈值（偏保守，保证“最稳”）
    min_traj_points: int = 200          # 一条轨迹至少这么多点才参与滑窗
    max_speed_mps: float = 18.0         # 约 35 knots
    max_step_jump_m: float = 600.0      # 每 30s 位移上限（极端点）
    min_sog_knots: float = 0.3          # 过滤掉长时间静止/漂移（可调）


    # 轨迹分段
    # 同一 MMSI 往往跨多天/多航次：这里先按时间 gap 把连续轨迹切成多段，再在“段内部”滑窗。
    # 当相邻两点时间间隔 > segment_gap_factor * dt（秒）时，认为开启新段。
    segment_gap_factor: float = 1.5


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _robust_angle_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return d


def load_ais_dataframe(path: str) -> pd.DataFrame:
    """
    读取 AIS 表格并标准化列名。
    必须包含：MMSI, Timestamp, LAT, LON, SOG, COG
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"data_path not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif ext in [".parquet"]:
        df = pd.read_parquet(path)
    elif ext in [".pkl", ".pickle"]:
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use CSV/Parquet/Pickle.")

    # 统一列名
    rename_map = {}
    for col in df.columns:
        c = col.strip()
        if c.lower() in ["mmsi", "ship_id", "id"]:
            rename_map[col] = "MMSI"
        elif c.lower() in ["timestamp", "time", "unix_time", "t"]:
            rename_map[col] = "Timestamp"
        elif c.lower() in ["lat", "latitude"]:
            rename_map[col] = "LAT"
        elif c.lower() in ["lon", "lng", "longitude"]:
            rename_map[col] = "LON"
        elif c.lower() in ["sog", "speed", "speed_over_ground"]:
            rename_map[col] = "SOG"
        elif c.lower() in ["cog", "course", "course_over_ground", "heading"]:
            rename_map[col] = "COG"
    df = df.rename(columns=rename_map)

    required = ["MMSI", "Timestamp", "LAT", "LON", "SOG", "COG"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column: {r}. Found: {list(df.columns)}")

    # 类型处理
    df["MMSI"] = df["MMSI"].astype(str)
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")
    df["COG"] = pd.to_numeric(df["COG"], errors="coerce")

    # 禁止静默 drop：一旦缺失/解析失败，直接报错
    na_mask = df[required].isna()
    if na_mask.any().any():
        counts = na_mask.sum().to_dict()
        raise ValueError(f"Found NaNs in required columns (will not auto-fill/drop): {counts}")
    return df


def build_local_projection_meta(df: pd.DataFrame) -> Dict:
    """全局使用同一个投影原点，保证不同船之间坐标一致。"""
    lat0 = float(df["LAT"].median())
    lon0 = float(df["LON"].median())
    return {
        "projection": "enu_wgs84",
        "wgs84_a": WGS84_A,
        "wgs84_f": WGS84_F,
        "wgs84_e2": WGS84_E2,
        "origin_lat": lat0,
        "origin_lon": lon0
    }


def run_step1_process(cfg: Step1Config) -> Dict:
    """
    生成训练用的数据文件：
      - x_train.npy   : (N, seq_len, 4)  [dx, dy, sog, cog]
      - y_offset.npy  : (N, pred_len, 2) [future dx, dy]
      - y_abs.npy     : (N, pred_len, 2) [future absolute x, y]
      - init_pos.npy  : (N, 2)           [anchor absolute x, y] at prediction start
      - ship_ids.npy  : (N,)             ship id per sample
      - segment_ids.npy : (N,)           同一 MMSI 的分段/航次 ID（用于 Split-C）
      - anchor_ts.npy   : (N,)           anchor 对应的 Timestamp（用于按时间切分）
      - geo_meta.json : 投影信息（可用于可视化还原经纬度）
      - data_config.json : step1 关键参数（供 step5 智能检测）
    """
    _ensure_dir(cfg.save_dir)

    logger.info("=" * 60)
    logger.info("[Step1] 读取 AIS 数据 + 清洗 + 切片 (Single-Ship)")
    logger.info(f"data_path: {cfg.data_path}")
    logger.info(f"save_dir : {cfg.save_dir}")
    logger.info(f"seq_len={cfg.seq_len}, pred_len={cfg.pred_len}, stride={cfg.stride}, dt={cfg.dt}s")
    logger.info("=" * 60)

    df = load_ais_dataframe(cfg.data_path)
    logger.info(f"原始记录数: {len(df)}  | 唯一 MMSI 数: {df['MMSI'].nunique()}")

    # 全局投影原点
    geo_meta = build_local_projection_meta(df)
    lat0, lon0 = geo_meta["origin_lat"], geo_meta["origin_lon"]
    with open(os.path.join(cfg.save_dir, "geo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(geo_meta, f, indent=2, ensure_ascii=False)

    # 逐船处理
    x_samples = []
    y_off_samples = []
    y_abs_samples = []
    init_pos_samples = []
    ship_id_samples = []
    segment_id_samples = []     # (N,) 每个样本属于哪个分段/航次
    anchor_ts_samples = []      # (N,) 每个样本 anchor 对应的时间戳（用于按时间切分）

    window_size = cfg.seq_len + cfg.pred_len
    n_seg_used = 0
    n_seg_skipped = 0
    n_windows = 0

    grouped = df.groupby("MMSI", sort=False)

    for mmsi, g in tqdm(grouped, desc="Step1|Ships"):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        if len(g) < window_size:
            continue

        ts_all = g["Timestamp"].to_numpy(np.float64)
        dt_steps_all = np.diff(ts_all)

        # 先按 gap 分段（避免一艘船跨天/跨航次造成“时间切分泄露”）
        seg_gap_sec = float(cfg.segment_gap_factor) * float(cfg.dt)
        break_idx = np.where(dt_steps_all > seg_gap_sec)[0]  # break between i and i+1
        seg_starts = [0] + [int(i + 1) for i in break_idx]
        seg_ends = [int(i) for i in break_idx] + [len(g) - 1]

        seg_counter = 0
        for seg_start, seg_end in zip(seg_starts, seg_ends):
            seg_len = seg_end - seg_start + 1
            if seg_len < max(cfg.min_traj_points, window_size + 5):
                n_seg_skipped += 1
                continue

            chunk_ranges = [(seg_start, seg_end)]

            for cs, ce in chunk_ranges:
                seg_counter += 1
                seg_id = f"{mmsi}__seg{seg_counter:04d}"

                lat = g["LAT"].iloc[cs : ce + 1].to_numpy(np.float64)
                lon = g["LON"].iloc[cs : ce + 1].to_numpy(np.float64)
                x, y = latlon_to_local_xy_m(lat, lon, lat0, lon0)

                sog_kn = g["SOG"].iloc[cs : ce + 1].to_numpy(np.float64)
                cog_deg = g["COG"].iloc[cs : ce + 1].to_numpy(np.float64)
                if np.any(np.isnan(sog_kn)) or np.any(np.isnan(cog_deg)):
                    raise ValueError(f"SOG/COG contains NaN for MMSI={mmsi} (segment={seg_id})")

                ts = ts_all[cs : ce + 1]
                dt_steps = np.diff(ts)
                # 允许少量抖动，比如 26~34s；如果该段重采样不一致，直接丢掉该段
                bad_dt = np.any((dt_steps < 0.5 * cfg.dt) | (dt_steps > 1.5 * cfg.dt))
                if bad_dt:
                    n_seg_skipped += 1
                    continue

                # ========== 清洗：步长跳变、速度异常 ==========
                dx = np.diff(x, prepend=x[0])
                dy = np.diff(y, prepend=y[0])
                step_dist = np.sqrt(dx**2 + dy**2)

                speed_mps = step_dist / max(float(cfg.dt), 1e-6)
                bad_speed = speed_mps > cfg.max_speed_mps

                bad_jump = step_dist > cfg.max_step_jump_m

                # 80% 分位速度过低，说明长时间静止/漂移，丢弃
                if np.nanpercentile(sog_kn, 80) < cfg.min_sog_knots:
                    n_seg_skipped += 1
                    continue

                if np.any(bad_speed) or np.any(bad_jump):
                    n_seg_skipped += 1
                    continue

                # ========== 构造 x_raw: [dx, dy, sog, cog] ==========
                x_raw = np.stack([dx, dy, sog_kn, cog_deg], axis=-1).astype(np.float32)  # (T,4)
                pos_abs = np.stack([x, y], axis=-1).astype(np.float32)                  # (T,2)

                # ========== 滑窗切片（在“段”内部切，避免跨段泄露）==========
                T = len(x_raw)
                if T < window_size:
                    n_seg_skipped += 1
                    continue

                used_any = False
                for s in range(0, T - window_size + 1, cfg.stride):
                    hist = x_raw[s : s + cfg.seq_len]                       # (seq,4)
                    fut_off = x_raw[s + cfg.seq_len : s + window_size, 0:2] # (pred,2) 未来 dx,dy
                    fut_abs = pos_abs[s + cfg.seq_len : s + window_size]    # (pred,2) 未来绝对
                    anchor = pos_abs[s + cfg.seq_len - 1]                   # (2,)
                    anchor_ts = float(ts[s + cfg.seq_len - 1])
                    # 过滤掉历史段长期静止/漂移的样本（预测起点如果处于停泊/极慢状态）
                    if float(np.nanmean(hist[-5:, 2])) < cfg.min_sog_knots:
                        continue

                    x_samples.append(hist)
                    y_off_samples.append(fut_off)
                    y_abs_samples.append(fut_abs)
                    init_pos_samples.append(anchor)
                    ship_id_samples.append(mmsi)
                    segment_id_samples.append(seg_id)
                    anchor_ts_samples.append(anchor_ts)
                    n_windows += 1
                    used_any = True

                if used_any:
                    n_seg_used += 1
                else:
                    n_seg_skipped += 1

    if n_windows == 0:
        raise RuntimeError("Step1 产生的样本数为 0。请检查数据路径、列名、清洗阈值、seq/pred/stride。")

    x_arr = np.stack(x_samples, axis=0)
    y_off_arr = np.stack(y_off_samples, axis=0)
    y_abs_arr = np.stack(y_abs_samples, axis=0)
    init_pos_arr = np.stack(init_pos_samples, axis=0)
    ship_ids_arr = np.array(ship_id_samples, dtype=object)
    segment_ids_arr = np.array(segment_id_samples, dtype=object)
    anchor_ts_arr = np.array(anchor_ts_samples, dtype=np.float64)

    np.save(os.path.join(cfg.save_dir, "x_train.npy"), x_arr)
    np.save(os.path.join(cfg.save_dir, "y_offset.npy"), y_off_arr)
    np.save(os.path.join(cfg.save_dir, "y_abs.npy"), y_abs_arr)
    np.save(os.path.join(cfg.save_dir, "init_pos.npy"), init_pos_arr)
    np.save(os.path.join(cfg.save_dir, "ship_ids.npy"), ship_ids_arr)
    np.save(os.path.join(cfg.save_dir, "segment_ids.npy"), segment_ids_arr)
    np.save(os.path.join(cfg.save_dir, "anchor_ts.npy"), anchor_ts_arr)

    # 保存 step1 参数，用于 step5 的智能检测
    data_cfg = {
        "data_path": cfg.data_path,
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "stride": cfg.stride,
        "window_size": window_size,
        "dt": cfg.dt,
        "max_speed_mps": cfg.max_speed_mps,
        "max_step_jump_m": cfg.max_step_jump_m,
        "min_sog_knots": cfg.min_sog_knots,
        "min_traj_points": cfg.min_traj_points,
        "segment_gap_factor": float(cfg.segment_gap_factor),
        # for Step6: restore lat/lon from local meters
        "geo_to_xy": {
            "enabled": True,
            "method": "enu_wgs84",
            "ref_lat_deg": lat0,
            "ref_lon_deg": lon0,
            "wgs84_a": WGS84_A,
            "wgs84_f": WGS84_F,
        },
    }
    with open(os.path.join(cfg.save_dir, "data_config.json"), "w", encoding="utf-8") as f:
        json.dump(data_cfg, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("[Step1] 完成 ✅")
    logger.info(f"有效轨迹分段数: {n_seg_used} | 丢弃分段数: {n_seg_skipped}")
    logger.info(f"生成样本数: {n_windows}")
    logger.info(f"x_train: {x_arr.shape} | y_offset: {y_off_arr.shape} | y_abs: {y_abs_arr.shape}")
    logger.info("=" * 60)

    return {
        "n_seg_used": n_seg_used,
        "n_seg_skipped": n_seg_skipped,
        "n_windows": n_windows,
        "save_dir": cfg.save_dir,
        "geo_meta": geo_meta,
    }


if __name__ == "__main__":
    # 允许单独运行 step1（读取环境变量或手改下面）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )

    # 示例：你可以直接改成你的路径
    cfg = Step1Config(
        data_path="ais.csv",
        save_dir="./exp_data_single",
        seq_len=40,
        pred_len=20,
        stride=10,
        dt=30.0,
    )
    run_step1_process(cfg)
