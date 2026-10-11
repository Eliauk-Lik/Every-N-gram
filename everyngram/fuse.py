"""S3m: 免训练融合。零机器学习依赖,只有向量运算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("arithmetic", "geometric")


@dataclass(frozen=True)
class Gate:
    """融合门槛:三个条件同时满足才融合,否则该位置退回 p_θ。

    分别对应三类"融合只会破坏"的位置:命中阶数太低(键太泛)、
    记忆分布不集中(噪声大)、模型本身已经很确定。
    """

    min_order: int = 1
    min_share: float = 0.0
    max_top_prob: float = 1.0

    def __post_init__(self) -> None:
        if int(self.min_order) < 1:
            raise ValueError(f"min_order 必须 >= 1: {self.min_order}")
        if not 0.0 <= float(self.min_share) <= 1.0:
            raise ValueError(f"min_share 必须在 [0, 1] 内: {self.min_share}")
        if not 0.0 < float(self.max_top_prob) <= 1.0:
            raise ValueError(f"max_top_prob 必须在 (0, 1] 内: {self.max_top_prob}")

    def allows(self, key, entry, top_prob: float) -> bool:
        """key/entry 来自记忆命中,top_prob 是模型分布的最大概率。"""
        if len(key) < int(self.min_order):
            return False
        total = sum(float(count) for _token, count in entry.next_top)
        if total <= 0:
            return False
        share = max(float(count) for _token, count in entry.next_top) / total
        if share < float(self.min_share):
            return False
        return float(top_prob) <= float(self.max_top_prob)

    def as_dict(self) -> dict:
        return {
            "min_order": int(self.min_order),
            "min_share": float(self.min_share),
            "max_top_prob": float(self.max_top_prob),
        }


@dataclass(frozen=True)
class FusionSpec:
    """一次融合的配置。memory 为 None 时不查表,等价于不融合。"""

    lam: float
    mode: str = "arithmetic"
    memory: object | None = None
    gate: Gate | None = None

    def __post_init__(self) -> None:
        value = float(self.lam)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"lam 必须在 [0, 1] 内: {self.lam}")
        if self.mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一: {self.mode}")


def memory_dist(entry, vocab_size: int) -> np.ndarray:
    """把 next_top 计数归一化成词表分布;无效输入返回全零。"""
    q = np.zeros(int(vocab_size), dtype=np.float64)
    if entry is None:
        return q
    for token, count in entry.next_top:
        index = int(token)
        if 0 <= index < q.shape[0] and float(count) > 0:
            q[index] = float(count)
    total = q.sum()
    if total <= 0:
        return np.zeros(int(vocab_size), dtype=np.float64)
    return q / total


def fuse_probs(p, q, lam: float, mode: str) -> np.ndarray:
    """融合模型分布 p 与记忆分布 q。

    算术融合的支撑集是并集;几何融合严格取交集(q 为零或 p 为零的位置输出零)。
    lam=0 或 q 全零时返回 p 的副本。
    """
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam 必须在 [0, 1] 内: {lam}")
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一: {mode}")

    p = np.asarray(p, dtype=np.float64)
    q = np.zeros_like(p) if q is None else np.asarray(q, dtype=np.float64)
    if lam <= 0.0 or q.sum() <= 0:
        return p.copy()

    if mode == "arithmetic":
        mixed = (1.0 - lam) * p + lam * q
    else:
        mask = (q > 0) & (p > 0)
        mixed = np.zeros_like(p)
        if mask.any():
            log_mixed = (1.0 - lam) * np.log(p[mask]) + lam * np.log(q[mask])
            log_mixed = log_mixed - log_mixed.max()
            mixed[mask] = np.exp(log_mixed)

    total = mixed.sum()
    if total <= 0:
        return p.copy()
    return mixed / total


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    """温度缩放的 softmax。温度下限 1e-6,避免除零。"""
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fused_next_probs(
    logits_row, ctx, fusion: FusionSpec | None, temperature: float = 1.0
) -> np.ndarray:
    """先 softmax,再按 fusion 查表融合。fusion 为 None 时只做 softmax。

    命中后还要过 gate:不放行就退回 p_θ,等价于该位置未命中。
    """
    p = softmax(logits_row, temperature)
    if fusion is None or fusion.memory is None:
        return p
    hit = fusion.memory.lookup(list(ctx))
    if hit is None:
        return p
    if fusion.gate is not None and not fusion.gate.allows(
        hit[0], hit[1], float(p.max())
    ):
        return p
    q = memory_dist(hit[1], p.shape[0])
    return fuse_probs(p, q, fusion.lam, fusion.mode)
