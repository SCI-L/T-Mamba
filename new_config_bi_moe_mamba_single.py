# new_config_bi_moe_mamba_single.py
# Edit values below and run Step5 with: --config new_config_bi_moe_mamba_single.py

CONFIG = {
    # -------------------------
    # [1] Data paths
    # -------------------------
    "data_path": r"D:\\American AIS Data\AIS_2025\\ais_2025_resampled_30s.csv",   # AIS 原始数据文件路径
    "save_dir": "./exp_data_single/mamba_bi_moe_V15",                       # Step1 预处理输出目录
    "res_dir": "./exp_results_single/mamba_bi_moe_V15",                     # 训练/评估输出目录

    # -------------------------
    # [2] Windowing (steps)
    # -------------------------
    "seq_len": 40,              # 历史步数
    "pred_len": 20,             # 预测步数
    "stride": 10,               # 滑窗步长(步)
    "dt": 30.0,                 # 采样间隔(秒)

    # -------------------------
    # [3] Step1 cleaning
    # -------------------------
    "min_traj_points": 200,         # 单条轨迹最少点数
    "max_speed_mps": 15.0,          # 最大速度(m/s)
    "max_step_jump_m": 600.0,       # 单步最大位移(米)
    "min_sog_knots": 0.3,           # 最小航速(knots)

    # [3.1] Step1 segmentation
    "segment_gap_factor": 1.5,      # 分段时间 gap 倍数

    # -------------------------
    # [4] DataLoader
    # -------------------------
    "batch_size": 256,              # 训练 batch
    "eval_batch_size": 256,         # 验证/测试 batch
    "num_workers": 4,               # DataLoader 进程数

    # [4.1] Split-C
    "test_ship_ratio": 0.2,         # Test MMSI 比例
    "val_time_ratio": 0.15,         # 非 Test 内时间切分比例
    "time_split_buffer_steps": 60,  # cutoff 缓冲步数

    # [4.2] Loss type
    "loss_type": "mixture",  # loss 类型: mixture

    # -------------------------
    # [5] Model
    # -------------------------
    "d_model": 256,             # 隐藏维度
    "n_layers": 6,              # 编码层数
    "dropout": 0.1,             # 丢弃率
    "d_state": 32,              # Mamba 状态维度
    "d_conv": 4,                # Mamba 卷积核宽度
    "expand": 2,                # Mamba expand 比例
    "d_ff": 1024,               # FFN 中间维度
    "aux_loss_coef": 0.01,      # MoE 辅助损失权重
    "gate_recent_window": 4,    # gate 最近窗口步数
    "use_init_pos": True,       # gate 是否使用初始绝对位置
    "dxdy_std_min": 1e-3,       # dx/dy 标准差下限
    "init_pos_std_min": 1e-3,   # 初始位置标准差下限

    # [6] Experts (K)
    "num_modes": 7,  # 专家/模式数(K)

    # [6.1] Viterbi decode
    "viterbi_max_switches": 3,      # 最大切换次数
    "viterbi_switch_cost": 0.1,     # 切换惩罚
    "viterbi_beam_size": 3,         # Beam Search 保留路径数(>1 启用)

    # [9] Step6 visualization
    "heatmap_grid": 260,  # 热力图网格分辨率

    # -------------------------
    # [7] Train
    # -------------------------
    "epochs": 50,                   # 训练轮数
    "lr": 0.0002,                   # 学习率
    "weight_decay": 0.01,           # 权重衰减
    "use_amp": True,                # 是否启用 AMP
    "grad_clip": 1.0,               # 梯度裁剪
    "lr_scheduler_factor": 0.5,     # 学习率衰减因子
    "lr_scheduler_patience": 5,     # 学习率衰减耐心
    "plot_every_epochs": 10,        # 画图间隔
    "seed": 42,                     # 随机种子

    # -------------------------
    # [8] Loss config
    # -------------------------
    "loss_cfg": {
        # loss_type=mixture
        "mixture_nll_weight": 1.0,   # soft mixture NLL 权重
        "load_balance": 0.2,         # gate 负载均衡
        "entropy": 0.02,             # gate 熵正则
        "gate_smooth_weight": 0.1,   # gate 平滑正则权重
        "entropy_decay_epochs": 0,   # 熵衰减轮数(0=不衰减)

        # shared
        "min_sigma_m": 2.0,         # sigma 下限(米)

        # physics consistency
        "phys_consistency_weight": 0.1,
        "phys_consistency_source": "target",  # target or prior
        "phys_consistency_loss": "huber",     # huber or l2
        "phys_consistency_huber_beta": 1.0,
    },
}
