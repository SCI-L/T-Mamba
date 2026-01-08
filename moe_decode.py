"""
moe_decode.py

给 Step5/Step6 用的 MoE 解码工具：

目标：把 per-step gate 概率 pi(t,k) 解码成一条“分段切换”的专家序列 k_t，
从而得到 **单条轨迹**。

核心：受限切换的 Viterbi（max_switches + switch_cost）。
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ViterbiResult:
    k_steps: np.ndarray  # (P,) int64
    n_switches: int
    score: float


def count_switches(k_steps: np.ndarray) -> int:
    k = np.asarray(k_steps).astype(np.int64).reshape(-1)
    if k.size <= 1:
        return 0
    return int(np.sum(k[1:] != k[:-1]))


def viterbi_decode_limited_switch(
    pi_step: np.ndarray,
    max_switches: int = 2,
    switch_cost: float = 1.0,
    eps: float = 1e-12,
) -> ViterbiResult:
    """
    受限切换 Viterbi：
      max_switches：最多允许切换次数（段数=切换+1）
      switch_cost ：每次切换的惩罚（越大越不愿意切换；相当于先验“段更平滑”）

    输入：
      pi_step: (P,K) 每个未来步 t 的 gate 概率分布（softmax 后）

    输出：
      k_steps: (P,) 每步采用的 expert/mode index（0..K-1）

    说明：
      我们最大化：
        sum_t log pi(t, k_t)  -  switch_cost * sum_t I[k_t != k_{t-1}]
      同时约束切换次数 <= max_switches。
    """
    pi_step = np.asarray(pi_step, dtype=np.float64)
    if pi_step.ndim != 2:
        raise ValueError(f"pi_step must be (P,K), got shape={pi_step.shape}")
    P, K = pi_step.shape
    if P <= 0 or K <= 0:
        raise ValueError(f"invalid pi_step shape={pi_step.shape}")

    S = int(max(0, max_switches))
    logp = np.log(pi_step + float(eps))  # (P,K)

    # dp[t, k, s] = best score up to step t (inclusive), end at expert k, using s switches.
    dp = np.full((P, K, S + 1), -np.inf, dtype=np.float64)
    prev_k = np.full((P, K, S + 1), -1, dtype=np.int16)
    prev_s = np.full((P, K, S + 1), -1, dtype=np.int16)

    # init at t=0: no switch
    dp[0, :, 0] = logp[0, :]
    prev_k[0, :, 0] = -1
    prev_s[0, :, 0] = -1

    for t in range(1, P):
        for s in range(0, S + 1):
            for k in range(K):
                # 1) stay with same expert: s unchanged
                best_score = dp[t - 1, k, s] + logp[t, k]
                best_pk = k
                best_ps = s

                # 2) switch from other expert: s decreases by 1
                if s > 0:
                    # try all previous experts
                    cand = dp[t - 1, :, s - 1] - float(switch_cost)  # (K,)
                    if K > 1:
                        cand = cand.copy()
                        cand[k] = -np.inf  # 真正切换：不允许从 k 切换到 k
                        cand_k = int(np.argmax(cand))
                        if np.isfinite(cand[cand_k]):
                            cand_score = cand[cand_k] + logp[t, k]
                            if cand_score > best_score:
                                best_score = cand_score
                                best_pk = cand_k
                                best_ps = s - 1

                dp[t, k, s] = best_score
                prev_k[t, k, s] = best_pk
                prev_s[t, k, s] = best_ps

    # best terminal among s<=S
    flat = int(np.argmax(dp[P - 1, :, :]))
    end_k, end_s = np.unravel_index(flat, (K, S + 1))
    best = float(dp[P - 1, end_k, end_s])

    # backtrack
    ks = np.zeros((P,), dtype=np.int64)
    k = end_k
    s = end_s
    for t in range(P - 1, -1, -1):
        ks[t] = k
        pk = int(prev_k[t, k, s])
        ps = int(prev_s[t, k, s])
        if t == 0:
            break
        k, s = pk, ps

    return ViterbiResult(k_steps=ks, n_switches=count_switches(ks), score=best)


def beam_decode_limited_switch(
    pi_step: np.ndarray,
    max_switches: int = 2,
    switch_cost: float = 1.0,
    beam_size: int = 3,
    eps: float = 1e-12,
) -> ViterbiResult:
    """
    Beam search decode with switch limit.
    Keeps top-N partial paths each step and applies the same switch_cost penalty.
    """
    pi_step = np.asarray(pi_step, dtype=np.float64)
    if pi_step.ndim != 2:
        raise ValueError(f"pi_step must be (P,K), got shape={pi_step.shape}")
    P, K = pi_step.shape
    if P <= 0 or K <= 0:
        raise ValueError(f"invalid pi_step shape={pi_step.shape}")

    beam_size = int(max(1, beam_size))
    if beam_size <= 1:
        return viterbi_decode_limited_switch(
            pi_step,
            max_switches=max_switches,
            switch_cost=switch_cost,
            eps=eps,
        )

    S = int(max(0, max_switches))
    logp = np.log(pi_step + float(eps))  # (P,K)

    # beam entries: (score, last_k, switches, path_list)
    beams = []
    for k in range(K):
        beams.append((float(logp[0, k]), int(k), 0, [int(k)]))
    beams.sort(key=lambda x: x[0], reverse=True)
    beams = beams[:beam_size]

    for t in range(1, P):
        candidates = []
        for score, k_prev, s_prev, path in beams:
            for k in range(K):
                s_new = s_prev + (1 if k != k_prev else 0)
                if s_new > S:
                    continue
                new_score = score + float(logp[t, k])
                if k != k_prev:
                    new_score -= float(switch_cost)
                candidates.append((new_score, int(k), s_new, path + [int(k)]))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

    best = max(beams, key=lambda x: x[0])
    ks = np.asarray(best[3], dtype=np.int64)
    return ViterbiResult(k_steps=ks, n_switches=count_switches(ks), score=float(best[0]))
