# new_step6_single_vis.py
"""
 Step6: 可视化 (Single-Ship, 单条轨迹)
- 读取 Step5 输出:
  - test_hist.npy (N, seq+1, 2)  history abs (meters)
  - test_true.npy (N, P, 2)      future gt abs (meters)
  - test_pred.npy (N, P, 2)      Top1 pred abs (meters)
  - test_dist_params.npy (N, P, 5) [x,y,sx,sy,rho] Top1
  - test_kinematics.npy (N, P, 2) [pred_sog(knots), pred_rot(deg/min)] Top1

  - （可选）per-step gate 解释性输出（用于“分段切换专家”检查）：
    - test_pi_steps.npy (N, P, K) 每个未来步的 gate 概率
    - test_k_steps.npy  (N, P)    Viterbi 解码得到的专家序列

用法:
  python new_step6_single_vis.py --res_dir ./exp_results_single --save_dir ./exp_data_single
或
  python new_step6_single_vis.py --config new_config_bi_moe_mamba_single.py

输出:
  ./exp_results_single/vis/
    - summary_errors.png
    - summary_ecdf.png
    - sample_XXXX.png (多张)
"""

import os
import json
import math
import random
import argparse
import logging
import importlib.util
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Optional

logger = logging.getLogger("Step6")


# ==========================================================
# 0) IO / Config
# ==========================================================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_cfg_dict_from_py(py_path: str):
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

    if not isinstance(d, dict):
        raise ValueError("Python config must return a dict.")
    return d


def load_cfg_dict(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        return _load_cfg_dict_from_py(path)
    return load_json(path)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def try_load(path: str):
    return np.load(path) if os.path.exists(path) else None


# ==========================================================
# 1) 坐标：米 <-> 经纬度（可选）
# Step1 会在 data_config.json 写入 geo_to_xy（method/ref_lat/ref_lon），这里按 method 选择反变换。
# ==========================================================
def xy_to_latlon_equirect(x_m, y_m, ref_lat_deg, ref_lon_deg, R=6378137.0):
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    lat0 = np.deg2rad(float(ref_lat_deg))
    lon0 = np.deg2rad(float(ref_lon_deg))

    lat = lat0 + (y / R)
    lat_mid = (lat + lat0) * 0.5
    lon = lon0 + (x / (R * np.cos(lat_mid) + 1e-12))

    return np.rad2deg(lat), np.rad2deg(lon)


# --- ENU (WGS84) reversible mapping ---
WGS84_A = 6378137.0
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


def xy_to_latlon_enu(x_m, y_m, ref_lat_deg, ref_lon_deg):
    """
    Local ENU meters -> WGS84 lat/lon (deg). Up=0.
    """
    e = np.asarray(x_m, dtype=np.float64)
    n = np.asarray(y_m, dtype=np.float64)
    u = np.zeros_like(e)

    lat0 = float(ref_lat_deg)
    lon0 = float(ref_lon_deg)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    slat, clat = np.sin(lat0_rad), np.cos(lat0_rad)
    slon, clon = np.sin(lon0_rad), np.cos(lon0_rad)

    # ENU -> ECEF delta
    dx = -slon * e - slat * clon * n + clat * clon * u
    dy = clon * e - slat * slon * n + clat * slon * u
    dz = clat * n + slat * u

    x0, y0, z0 = _geodetic_to_ecef(np.array([lat0]), np.array([lon0]), h_m=0.0)
    x_ecef = dx + float(x0[0])
    y_ecef = dy + float(y0[0])
    z_ecef = dz + float(z0[0])

    lat_deg, lon_deg, _h = _ecef_to_geodetic(x_ecef, y_ecef, z_ecef)
    return lat_deg, lon_deg


def maybe_get_geo_ref(save_dir: str):
    """
    尝试从 Step1 的 data_config.json 中读取 ref_lat/ref_lon
    """
    cfg_path = os.path.join(save_dir, "data_config.json")
    if not os.path.exists(cfg_path):
        return None
    cfg = load_json(cfg_path)
    geo = cfg.get("geo_to_xy", None)
    if not isinstance(geo, dict):
        return None
    if not geo.get("enabled", False):
        return None
    ref_lat = geo.get("ref_lat_deg", None)
    ref_lon = geo.get("ref_lon_deg", None)
    method = geo.get("method", "equirectangular")
    R = geo.get("R", 6378137.0)
    if ref_lat is None or ref_lon is None:
        return None
    return {
        "method": str(method),
        "ref_lat_deg": float(ref_lat),
        "ref_lon_deg": float(ref_lon),
        "R": float(R),
    }


# ==========================================================
# 2) 指标
# ==========================================================
def euclid_errors(pred_abs: np.ndarray, true_abs: np.ndarray):
    """
    pred_abs: (N,P,2)
    true_abs: (N,P,2)
    return: (N,P)
    """
    return np.linalg.norm(pred_abs - true_abs, axis=-1)


def ecdf(arr: np.ndarray):
    xs = np.sort(arr)
    ys = np.arange(1, len(xs) + 1) / max(len(xs), 1)
    return xs, ys


def unwrap_deg(deg: np.ndarray):
    """deg: (...,T) -> unwrap for plotting"""
    rad = np.deg2rad(deg)
    rad_u = np.unwrap(rad, axis=-1)
    return np.rad2deg(rad_u)


def kinematics_from_abs(traj_abs: np.ndarray, dt: float):
    """
    traj_abs: (T,2) 绝对坐标（米）
    return:
      sog_knots: (T-1,)
      cog_deg:   (T-1,)
      rot_deg_min: (T-1,) 角速度 deg/min（用 unwrapped COG）
    """
    dif = traj_abs[1:] - traj_abs[:-1]
    dist = np.linalg.norm(dif, axis=-1)
    sog_knots = (dist / dt) * 1.9438444924406

    cog = np.degrees(np.arctan2(dif[:, 1], dif[:, 0]))
    cog = (cog + 360.0) % 360.0
    cog_u = unwrap_deg(cog[None, :])[0]
    rot = np.zeros_like(cog_u)
    rot[1:] = cog_u[1:] - cog_u[:-1]
    rot_deg_min = rot * (60.0 / dt)
    return sog_knots, cog, rot_deg_min


# ==========================================================
# 3) 不确定性椭圆
# ==========================================================
def add_cov_ellipse(ax, mean_xy, sx, sy, rho, n_std=2.0, **kwargs):
    """
    mean_xy: (2,)
    sx, sy: std (meters)
    rho: correlation
    """
    sx = float(max(sx, 1e-6))
    sy = float(max(sy, 1e-6))
    r = float(np.clip(rho, -0.999, 0.999))

    cov = np.array([[sx * sx, r * sx * sy],
                    [r * sx * sy, sy * sy]], dtype=np.float64)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width = 2.0 * n_std * math.sqrt(max(vals[0], 1e-12))
    height = 2.0 * n_std * math.sqrt(max(vals[1], 1e-12))

    e = Ellipse(xy=mean_xy, width=width, height=height, angle=angle, fill=False, **kwargs)
    ax.add_patch(e)


# ==========================================================
# 3.1) 热力图（按每个未来步叠加密度）
# ==========================================================
def _bivar_gaussian_pdf_grid(X, Y, mx, my, sx, sy, rho):
    """
    X,Y: (H,W) grid in meters
    mx,my: scalars (meters)
    sx,sy: scalars (meters, >0)
    rho: scalar in [-1,1]
    return: (H,W) pdf (meters^-2)
    """
    sx = float(max(sx, 1e-6))
    sy = float(max(sy, 1e-6))
    r = float(np.clip(rho, -0.999, 0.999))
    one = max(1.0 - r * r, 1e-6)

    dx = X - float(mx)
    dy = Y - float(my)

    a = 1.0 / (sx * sx * one)
    b = -r / (sx * sy * one)
    c = 1.0 / (sy * sy * one)

    expo = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)
    norm = 1.0 / (2.0 * math.pi * sx * sy * math.sqrt(one))
    return norm * np.exp(expo)


def add_topk_density_heatmap(ax,
                            hist_xy: Optional[np.ndarray],
                            true_xy: np.ndarray,
                            params_modes: np.ndarray,
                            pi_k: np.ndarray,
                            topk: int = 3,
                            grid_size: int = 260,
                            pad_ratio: float = 0.12,
                            cmap: str = "turbo",
                            min_density_ratio: float = 0.02):
    """
    在 XY 图上渲染“连续热斑”：对 Top-K modes 的每个未来步二维高斯密度求和。

    hist_xy: (seq+1,2)
    true_xy: (P,2)
    params_modes: (K,P,5)  [x,y,sx,sy,rho]（绝对坐标，单位米）
    pi_k: (K,) gate 权重
    """
    if params_modes is None or pi_k is None:
        return None, None

    K, P, D = params_modes.shape
    if K == 0 or P == 0:
        return None, None

    topk = int(min(max(int(topk), 1), K))
    order = np.argsort(-pi_k)
    sel = order[:topk]
    w = pi_k[sel].astype(np.float64)
    w = w / (w.sum() + 1e-12)

    # 取范围：GT + TopK modes（历史可选）
    pts = [true_xy]
    if hist_xy is not None:
        pts.append(hist_xy)
    for k in sel:
        pts.append(params_modes[k, :, 0:2])
    pts = np.concatenate(pts, axis=0)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    dx = float(x_max - x_min)
    dy = float(y_max - y_min)
    pad = max(dx, dy) * float(pad_ratio) + 1.0
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    xs = np.linspace(x_min, x_max, int(grid_size))
    ys = np.linspace(y_min, y_max, int(grid_size))
    X, Y = np.meshgrid(xs, ys)

    dens = np.zeros_like(X, dtype=np.float64)

    # 叠加：sum_{k in topK} w_k * sum_{t} N(x | mu_{k,t}, Sigma_{k,t})
    for kk, k in enumerate(sel):
        wk = float(w[kk])
        for t in range(P):
            mx, my, sx, sy, rho = params_modes[k, t]
            dens += wk * _bivar_gaussian_pdf_grid(X, Y, mx, my, sx, sy, rho)

    # 归一化到可视化友好（不改变相对形状）
    vmax = float(np.percentile(dens, 99.5)) if np.any(dens > 0) else 1.0
    vmax = max(float(vmax), 1e-12)

    # 低密度区域透明：避免 imshow 显示成一大块矩形“底图”
    thr = float(vmax) * float(max(min_density_ratio, 0.0))
    dens_ma = np.ma.masked_less(dens, thr)

    try:
        cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    except Exception:
        cmap_obj = matplotlib.cm.get_cmap(cmap)
    try:
        cmap_obj = cmap_obj.copy()
    except Exception:
        pass
    try:
        cmap_obj.set_bad((1.0, 1.0, 1.0, 0.0))  # transparent for masked
    except Exception:
        pass

    im = ax.imshow(
        dens_ma,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap=cmap_obj,
        alpha=0.85,
        vmin=0.0,
        vmax=vmax,
        interpolation="bilinear",
        aspect="auto",
    )
    return im, {"topk": topk, "sel": sel, "weights": w, "extent": [x_min, x_max, y_min, y_max], "vmax": vmax}

# ==========================================================
# 4) 全局统计图
# ==========================================================
def plot_global_summaries(vis_dir: str,
                          hist: np.ndarray,
                          true: np.ndarray,
                          pred_top1: np.ndarray,
                          dist_top1: np.ndarray,
                          ):
    ensure_dir(vis_dir)

    err_top = euclid_errors(pred_top1, true)
    ade_top = err_top.mean(axis=1)
    fde_top = err_top[:, -1]

    # Top1 不确定性（平均 sigma）
    sigma = dist_top1[..., 2:4]
    unc = np.mean(np.sqrt(sigma[..., 0] ** 2 + sigma[..., 1] ** 2), axis=1)

    # 1) summary_errors
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.hist(ade_top, bins=40)
    ax1.set_title("Top1 ADE distribution (m)")
    ax1.set_xlabel("ADE (m)")
    ax1.set_ylabel("count")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(fde_top, bins=40)
    ax2.set_title("Top1 FDE distribution (m)")
    ax2.set_xlabel("FDE (m)")
    ax2.set_ylabel("count")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(unc, ade_top, s=10, alpha=0.6)
    ax3.set_title("Top1 Uncertainty vs ADE")
    ax3.set_xlabel("mean sigma (m)")
    ax3.set_ylabel("ADE (m)")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "summary_errors.png"), dpi=160)
    plt.close(fig)

    # 2) ECDF
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    xs, ys = ecdf(ade_top)
    ax.plot(xs, ys, label="Top1 ADE")
    xs, ys = ecdf(fde_top)
    ax.plot(xs, ys, linestyle="--", label="Top1 FDE")

    ax.set_title(f"ECDF (N={len(ade_top)})")
    ax.set_xlabel("Error threshold (m)")
    ax.set_ylabel("Proportion (0-1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "summary_ecdf.png"), dpi=160)
    plt.close(fig)

    logger.info(f"[Step6] 全局图已保存: {vis_dir}/summary_errors.png, summary_ecdf.png")


# ==========================================================
# 5) 单样本可视化
# ==========================================================
def plot_one_sample(vis_dir: str,
                    idx: int,
                    dt: float,
                    hist: np.ndarray,
                    true: np.ndarray,
                    pred_top1: np.ndarray,
                    dist_top1: np.ndarray,
                    kin_top1: np.ndarray,
                    pi_steps: np.ndarray = None,
                    k_steps: np.ndarray = None,
                    heatmap_grid: int = 260,
                    heatmap_enable: bool = True,
                    show_xy_colorbar: bool = False,
                    show_pi_bar: bool = False,
                    ellipse_steps=(5, 10, 20),
                    geo_ref=None):
    ensure_dir(vis_dir)

    h = hist[idx]               # (seq+1,2)
    t = true[idx]               # (P,2)
    p1 = pred_top1[idx]         # (P,2)
    params1 = dist_top1[idx]    # (P,5)
    kin1 = kin_top1[idx]        # (P,2)
    anchor = h[-1]

    P = t.shape[0]
    time_min = np.arange(1, P + 1) * (dt / 60.0)

    # 误差曲线：Top1
    err1 = np.linalg.norm(p1 - t, axis=-1)

    # 计算 GT 动力学（基于 anchor + true）
    gt_full = np.vstack([anchor[None, :], t])
    gt_sog, gt_cog, gt_rot = kinematics_from_abs(gt_full, dt=dt)
    # pred 动力学（基于 anchor + top1）
    p_full = np.vstack([anchor[None, :], p1])
    p_sog, p_cog, p_rot = kinematics_from_abs(p_full, dt=dt)

    # 画图布局：2x2
    # 用 constrained_layout 避免标题/边距互相挤压
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)

    # (1) Trajectory XY（Top-3 热力图 + 轨迹线）
    ax = fig.add_subplot(2, 2, 1)

    # 单条轨迹的热力图：默认使用 dist_top1（即 Step5 解码后的单条轨迹分布）
    # 只渲染单条轨迹的热力图（来自 dist_top1）
    heat_im = None
    heat_info = None
    if bool(heatmap_enable):
        try:
            params_k = params1[None, :, :]  # (1,P,5)
            pik = np.array([1.0], dtype=np.float64)
            heat_im, heat_info = add_topk_density_heatmap(
                ax,
                hist_xy=None,
                true_xy=t,
                params_modes=params_k,
                pi_k=pik,
                topk=1,
                grid_size=int(heatmap_grid),
                pad_ratio=0.12,
                cmap="turbo",
                min_density_ratio=0.02,
            )
        except Exception as e:
            logger.warning(f"[Step6] heatmap failed for idx={idx}: {e}")
            heat_im = None

    # 这里不再画历史轨迹：历史已在 *_latlon.png 中展示，避免左上角拥挤
    ax.plot(t[:, 0], t[:, 1], linewidth=2.2, color="#2ca02c", label="GT (future)")
    ax.plot(p1[:, 0], p1[:, 1], linewidth=2.2, linestyle="--", color="#1f77b4", label="Top1 pred")
    ax.scatter(anchor[0], anchor[1], s=70, marker="*", color="#111111", label="Anchor")

    # 不再绘制 Top2 / all-modes：你现在要的是“单条轨迹+单模态热力图”

    # 椭圆：Top1
    for s in ellipse_steps:
        if 1 <= s <= P:
            mu_xy = params1[s - 1, 0:2]
            sx, sy = params1[s - 1, 2], params1[s - 1, 3]
            rho = params1[s - 1, 4]
            add_cov_ellipse(ax, mu_xy, sx, sy, rho, n_std=2.0, linewidth=1.2, alpha=0.6)

    ade = float(err1.mean())
    fde = float(err1[-1])
    sup_parts = [f"Sample {idx}", f"ADE={ade:.1f}m", f"FDE={fde:.1f}m"]
    fig.suptitle(" | ".join(sup_parts), fontsize=12)
    ax.set_title("Trajectory (XY)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    # 使用 adjustable="box" 避免 Matplotlib 在固定 x/y limits 下报
    # “Ignoring fixed x/y limits ... adjustable data limits” 的 warning。
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # 可选：在 XY 图里嵌入 gate 权重条形图（用 pi_steps 在时间维做均值）
    if bool(show_pi_bar) and (pi_steps is not None):
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            pik = pi_steps[idx].mean(axis=0).astype(np.float64)  # (K,)
            iax = inset_axes(ax, width="32%", height="22%", loc="upper right", borderpad=1.0)
            xs = np.arange(len(pik))
            iax.bar(xs, pik, color="#999999", alpha=0.85)
            if len(pik) > 0:
                k1 = int(np.argmax(pik))
                iax.bar([k1], [pik[k1]], color="#1f77b4", alpha=0.95)
            iax.set_ylim(0.0, float(max(0.35, pik.max() * 1.15 if pik.size else 1.0)))
            iax.set_xticks(xs)
            iax.set_xticklabels([str(int(i)) for i in xs], fontsize=7)
            iax.set_yticks([])
            iax.set_title("pi (gate)", fontsize=8)
            iax.grid(True, axis="y", alpha=0.15)
        except Exception:
            pass

    # 额外：保存“每步 gate 权重”图（单独一张，便于检查专家是否真的在分段切换）
    if (pi_steps is not None) and (k_steps is not None):
        try:
            ps = pi_steps[idx]  # (P,K)
            ks = k_steps[idx]   # (P,)
            _plot_gate_steps(vis_dir, idx, dt, ps, ks)
        except Exception as e:
            logger.warning(f"[Step6] gate-step plot failed for idx={idx}: {e}")

    # 聚焦视野：只围绕未来预测段放大（不看历史）
    zoom_pts = [t, p1]
    zoom_pts.append(anchor[None, :])
    zoom_pts = np.concatenate(zoom_pts, axis=0)
    zx_min, zy_min = zoom_pts.min(axis=0)
    zx_max, zy_max = zoom_pts.max(axis=0)
    zdx = float(zx_max - zx_min)
    zdy = float(zy_max - zy_min)
    pad = max(zdx, zdy) * 0.15 + 10.0
    zx_min -= pad
    zx_max += pad
    zy_min -= pad
    zy_max += pad
    # 让显示范围接近平方
    cx = 0.5 * (zx_min + zx_max)
    cy = 0.5 * (zy_min + zy_max)
    span = max(zx_max - zx_min, zy_max - zy_min)
    half = 0.5 * span
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    # 默认不在单样本图里画 colorbar：否则会挤压左上角子图宽度，导致和其他三幅不一致
    if heat_im is not None and bool(show_xy_colorbar):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax, width="3.0%", height="75%", loc="upper right", borderpad=1.2)
        cbar = fig.colorbar(heat_im, cax=cax)
        cbar.set_label("Density")

    # (2) Course over horizon (COG) - Top1 only
    ax2 = fig.add_subplot(2, 2, 2)
    def _unwrap_deg(deg_arr: np.ndarray) -> np.ndarray:
        rad = np.deg2rad(deg_arr.astype(np.float64))
        rad_u = np.unwrap(rad)
        return np.rad2deg(rad_u)

    gt_cog_u = _unwrap_deg(gt_cog)
    p_cog_u = _unwrap_deg(p_cog)
    ax2.plot(time_min, gt_cog_u, label="GT COG (deg)")
    ax2.plot(time_min, p_cog_u, linestyle="--", label="Top1 COG (deg)")
    ax2.set_title("Course (COG)")
    ax2.set_xlabel("Minutes into future")
    ax2.set_ylabel("deg (unwrapped)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # (3) Speed
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(time_min, gt_sog, label="GT SOG (knots)")
    ax3.plot(time_min, p_sog, linestyle="--", label="Top1 SOG (knots)")
    ax3.set_title("Speed (SOG)")
    ax3.set_xlabel("Minutes into future")
    ax3.set_ylabel("knots")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # (4) Course / Rotation
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(time_min, gt_rot, label="GT ROT (deg/min)")
    ax4.plot(time_min, p_rot, linestyle="--", label="Top1 ROT (deg/min)")
    ax4.set_title("Rotation (ROT)")
    ax4.set_xlabel("Minutes into future")
    ax4.set_ylabel("deg/min")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 单样本图使用 constrained_layout；不要再 tight_layout（会导致布局变化/警告）
    out_path = os.path.join(vis_dir, f"sample_{idx:05d}.png")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    # 可选：经纬度版本（如果能拿到 ref_lat/ref_lon）
    if geo_ref is not None:
        method = geo_ref.get("method", "equirectangular")
        ref_lat = geo_ref["ref_lat_deg"]
        ref_lon = geo_ref["ref_lon_deg"]

        if method == "enu_wgs84":
            lat_h, lon_h = xy_to_latlon_enu(h[:, 0], h[:, 1], ref_lat, ref_lon)
            lat_t, lon_t = xy_to_latlon_enu(t[:, 0], t[:, 1], ref_lat, ref_lon)
            lat_p, lon_p = xy_to_latlon_enu(p1[:, 0], p1[:, 1], ref_lat, ref_lon)
        else:
            R = float(geo_ref.get("R", 6378137.0))
            lat_h, lon_h = xy_to_latlon_equirect(h[:, 0], h[:, 1], ref_lat, ref_lon, R)
            lat_t, lon_t = xy_to_latlon_equirect(t[:, 0], t[:, 1], ref_lat, ref_lon, R)
            lat_p, lon_p = xy_to_latlon_equirect(p1[:, 0], p1[:, 1], ref_lat, ref_lon, R)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(lon_h, lat_h, marker=".", linestyle="--", alpha=0.6, label="History")
        ax.plot(lon_t, lat_t, linewidth=2.0, label="GT (future)")
        ax.plot(lon_p, lat_p, linewidth=2.0, linestyle="--", label="Top1 pred")
        ax.set_title(f"Lat/Lon view | sample {idx}")
        ax.set_xlabel("Lon (deg)")
        ax.set_ylabel("Lat (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path2 = os.path.join(vis_dir, f"sample_{idx:05d}_latlon.png")
        fig.savefig(out_path2, dpi=170)
        plt.close(fig)


def _plot_gate_steps(vis_dir: str, idx: int, dt: float, pi_step: np.ndarray, k_steps: np.ndarray):
    """
    单独输出一张“每个未来步的 gate 权重”图：
      - 颜色：pi(t,k)
      - 叠加：解码得到的专家序列 k_t（折线）
    """
    ensure_dir(vis_dir)
    pi_step = np.asarray(pi_step, dtype=np.float64)  # (P,K)
    k_steps = np.asarray(k_steps, dtype=np.int64).reshape(-1)  # (P,)
    P, K = pi_step.shape
    time_min = np.arange(1, P + 1) * (dt / 60.0)

    fig = plt.figure(figsize=(10, 3.8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        pi_step.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        extent=[time_min[0], time_min[-1], -0.5, K - 0.5],
        vmin=0.0,
        vmax=float(max(1e-6, np.max(pi_step))),
    )
    ax.plot(time_min, k_steps.astype(np.float64), color="#ff7f0e", linewidth=2.0, label="Decoded expert")
    ax.set_title(f"Gate weights per step | sample {idx}")
    ax.set_xlabel("Minutes into future")
    ax.set_ylabel("expert index")
    ax.set_yticks(list(range(K)))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", framealpha=0.9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("pi(t,k)")

    out_path = os.path.join(vis_dir, f"sample_{idx:05d}_gate.png")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ==========================================================
# 6) main
# ==========================================================
def main():
    # 便捷：如果用户未显式传 --config，但目录下存在默认配置文件，则自动使用它（与 Step5 保持一致）
    default_cfg = None
    for cand in ["new_config_bi_moe_mamba_single.py", "new_config_bi_moe_mamba_single.json"]:
        if os.path.exists(cand):
            default_cfg = cand
            break

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_cfg, help="config py/json (optional)")
    parser.add_argument("--res_dir", type=str, default="./exp_results_single")
    parser.add_argument("--save_dir", type=str, default="./exp_data_single")
    parser.add_argument("--dt", type=float, default=30.0)
    parser.add_argument("--num_vis", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--idx", type=int, default=None, help="指定某个样本 index（不随机）")
    parser.add_argument("--ellipse_steps", type=str, default="5,10,20", help="画不确定性椭圆的步号(1..P)，逗号分隔")
    parser.add_argument("--heatmap_grid", type=int, default=260, help="热力图网格分辨率（越大越细，但更慢）")
    parser.add_argument("--no_heatmap", action="store_true", help="关闭热力图渲染（左上角更干净）")
    parser.add_argument("--show_xy_colorbar", action="store_true", help="在单样本左上角 XY 图里显示 colorbar（默认不显示，避免挤压子图）")
    parser.add_argument("--show_pi_bar", action="store_true", help="在左上角嵌入 gate 权重条形图（对未来步取均值）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )

    # 若提供 config，优先用 config 的 res_dir/save_dir/dt（如果有）
    # 但如果用户显式传了 --res_dir/--save_dir/--dt/--heatmap_grid，则以命令行为准（避免误覆盖）。
    if args.config is not None and os.path.exists(args.config):
        cfg = load_cfg_dict(args.config)
        if args.res_dir == "./exp_results_single":
            args.res_dir = cfg.get("res_dir", args.res_dir)
        if args.save_dir == "./exp_data_single":
            args.save_dir = cfg.get("save_dir", args.save_dir)
        if args.dt == 30.0:
            args.dt = float(cfg.get("dt", args.dt))
        if args.heatmap_grid == 260:
            args.heatmap_grid = int(cfg.get("heatmap_grid", args.heatmap_grid))

    logger.info("=" * 60)
    logger.info("[Step6] Visualization (Single-Ship, Single Trajectory)")
    logger.info(f"config  : {args.config}")
    logger.info(f"res_dir : {args.res_dir}")
    logger.info(f"save_dir: {args.save_dir}")
    logger.info(f"dt      : {args.dt}")
    logger.info("=" * 60)

    vis_dir = os.path.join(args.res_dir, "vis")
    ensure_dir(vis_dir)

    # load required
    hist = np.load(os.path.join(args.res_dir, "test_hist.npy"))
    true = np.load(os.path.join(args.res_dir, "test_true.npy"))
    pred = np.load(os.path.join(args.res_dir, "test_pred.npy"))
    dist = np.load(os.path.join(args.res_dir, "test_dist_params.npy"))
    kin = np.load(os.path.join(args.res_dir, "test_kinematics.npy"))

    # per-step gate (optional)
    pi_steps = try_load(os.path.join(args.res_dir, "test_pi_steps.npy"))   # (N,P,K)
    k_steps = try_load(os.path.join(args.res_dir, "test_k_steps.npy"))     # (N,P)

    # geo ref (optional)
    geo_ref = maybe_get_geo_ref(args.save_dir)
    if geo_ref is not None:
        logger.info(
            f"[Step6] 检测到 geo ref: method={geo_ref.get('method')}, "
            f"ref_lat={geo_ref.get('ref_lat_deg'):.6f}, ref_lon={geo_ref.get('ref_lon_deg'):.6f}，将额外输出 lat/lon 图。"
        )
    else:
        logger.info("[Step6] 未检测到 geo ref（或 geo_to_xy 未启用），将只输出米坐标图。")

    # global plots
    plot_global_summaries(vis_dir, hist, true, pred, dist)

    # pick samples
    random.seed(args.seed)
    N = hist.shape[0]
    if args.idx is not None:
        indices = [int(args.idx)]
    else:
        n = min(args.num_vis, N)
        indices = random.sample(range(N), n)

    steps = []
    try:
        steps = [int(s.strip()) for s in args.ellipse_steps.split(",") if s.strip()]
    except Exception:
        steps = [5, 10, 20]

    logger.info(f"[Step6] 正在生成单样本图: {len(indices)} 张 -> {vis_dir}")
    heatmap_on = (not bool(args.no_heatmap))
    for idx in indices:
        plot_one_sample(
            vis_dir=vis_dir,
            idx=idx,
            dt=args.dt,
            hist=hist,
            true=true,
            pred_top1=pred,
            dist_top1=dist,
            kin_top1=kin,
            pi_steps=pi_steps,
            k_steps=k_steps,
            heatmap_grid=args.heatmap_grid,
            heatmap_enable=heatmap_on,
            show_xy_colorbar=bool(args.show_xy_colorbar),
            show_pi_bar=bool(args.show_pi_bar),
            ellipse_steps=tuple(steps),
            geo_ref=geo_ref
        )

    logger.info(f"[Step6] Done ✅ 可视化已生成: {vis_dir}")


if __name__ == "__main__":
    main()
