"""内容选择:C1 双池实验的载体。只依赖 memory 与 fuse 的向量运算。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np

from everyngram.fuse import memory_dist, softmax


def collect_contexts(
    files: Sequence[str | Path],
    tokenizer,
    *,
    n: int = 256,
    context_len: int = 128,
    horizon: int = 1,
    seed: int = 0,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """从文件里按固定种子抽 n 个 (上下文, 真实下一个 token)。"""
    rng = random.Random(seed)
    contexts: list[tuple[tuple[int, ...], int]] = []
    for path in files:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        ids = tokenizer.encode(text, add_special_tokens=False)
        span = len(ids) - context_len - horizon
        if span < 1:
            continue
        for start in rng.sample(range(span + 1), min(n - len(contexts), span + 1)):
            contexts.append(
                (
                    tuple(ids[start : start + context_len]),
                    int(ids[start + context_len]),
                )
            )
            if len(contexts) >= n:
                return tuple(contexts)
    return tuple(contexts)


def score_contexts(model, memory, contexts) -> list[dict]:
    """给每个上下文打两个分:速度分(草稿被接受的概率)与质量分(似然增益)。

    速度分 = p_θ[草稿首 token]:贪婪通路上草稿第一步不被拒的概率。
    质量分 = log q(真值) - log p_θ(真值):λ=1 口径下记忆在真值上的对数似然增益。
    """
    scored: list[dict] = []
    for prompt, true_next in contexts:
        logits, _ = model.forward(list(prompt))
        p = softmax(logits[-1])
        hit = memory.lookup(list(prompt)) if memory is not None else None
        entry = hit[1] if hit is not None else None
        item = {
            "key": hit[0] if hit is not None else None,
            "true_next": int(true_next),
            "draft_first": None,
            "speed_score": None,
            "quality_score": None,
        }
        if entry is not None and entry.continuation and 0 <= true_next < p.shape[0]:
            q = memory_dist(entry, p.shape[0])
            draft = int(entry.continuation[0])
            item["draft_first"] = draft
            item["speed_score"] = float(p[draft])
            if q.sum() > 0 and q[true_next] > 0 and p[true_next] > 0:
                item["quality_score"] = float(
                    np.log(q[true_next]) - np.log(p[true_next])
                )
        scored.append(item)
    return scored


def select_pools(scored: Sequence[dict], k: int) -> dict:
    """按两个分数各取 top-k,返回两个池与交叠统计(按 key 的交并比)。"""
    candidates = [
        item
        for item in scored
        if item["key"] is not None
        and item["speed_score"] is not None
        and item["quality_score"] is not None
    ]
    if not candidates:
        raise ValueError("没有可用候选:所有上下文的分数都缺失")
    k = int(k)
    by_speed = sorted(candidates, key=lambda item: item["speed_score"], reverse=True)[:k]
    by_quality = sorted(
        candidates, key=lambda item: item["quality_score"], reverse=True
    )[:k]
    speed_keys = {item["key"] for item in by_speed}
    quality_keys = {item["key"] for item in by_quality}
    union = speed_keys | quality_keys
    return {
        "speed_pool": by_speed,
        "quality_pool": by_quality,
        "overlap_count": len(speed_keys & quality_keys),
        "overlap": (len(speed_keys & quality_keys) / len(union)) if union else 0.0,
        "k_effective": k,
    }


def effective_k(scored: Sequence[dict], k: int) -> int:
    """把请求的 k 收敛到候选数的一半以内。

    k 大于等于候选数时两个池会退化成同一个集合(全部候选),交叠率恒为 1,
    读数失去意义;候选稀少时至少取 1,保证池非空。
    """
    candidates = [
        item
        for item in scored
        if item["key"] is not None
        and item["speed_score"] is not None
        and item["quality_score"] is not None
    ]
    if not candidates:
        raise ValueError("没有可用候选:所有上下文的分数都缺失")
    return max(1, min(int(k), len(candidates) // 2))


def overlap_curve(scored: Sequence[dict], ks) -> dict:
    """交叠率随 k 的曲线。

    单个 k 的交叠率会被饱和效应污染:k 接近候选数时两个池必然覆盖大部分
    候选,交叠率趋于 1。按 k 出曲线才能看出"在最前面几名上是否重合"。
    """
    candidates = [
        item
        for item in scored
        if item["key"] is not None
        and item["speed_score"] is not None
        and item["quality_score"] is not None
    ]
    if not candidates:
        raise ValueError("没有可用候选:所有上下文的分数都缺失")
    by_speed = sorted(candidates, key=lambda item: item["speed_score"], reverse=True)
    by_quality = sorted(
        candidates, key=lambda item: item["quality_score"], reverse=True
    )
    ks = [max(1, int(k)) for k in ks]
    overlaps: list[float] = []
    intersections: list[int] = []
    for k in ks:
        a = {item["key"] for item in by_speed[:k]}
        b = {item["key"] for item in by_quality[:k]}
        union = a | b
        intersections.append(len(a & b))
        overlaps.append(round(len(a & b) / len(union), 4) if union else 0.0)
    return {"ks": ks, "overlap": overlaps, "intersection": intersections}


def select_gate(model, memory, cases, *, gates, lambdas, min_coverage: float = 0.25) -> dict:
    """在 dev 用例上选融合门槛与 λ:覆盖率达标里取增益最大者。

    一次遍历完成:每个位置只算一次 p 与 q,再对每个 λ 算融合 NLL,
    累加到所有通过该位置的门对应的配置上。
    """
    from everyngram.fuse import fuse_probs

    gates = tuple(gates)
    lambdas = tuple(float(lam) for lam in lambdas)
    if not gates:
        raise ValueError("gates 不能为空")
    if not lambdas:
        raise ValueError("lambdas 不能为空")

    gains = [[0.0] * len(lambdas) for _ in gates]
    fired = [0] * len(gates)
    positions = 0

    for case in cases:
        prompt = list(case.prompt_ids)
        reference = list(case.reference_ids)
        if not reference:
            continue
        logits, _ = model.forward(prompt + reference[:-1])
        ctx = list(prompt)
        for i, token in enumerate(reference):
            positions += 1
            p = softmax(logits[len(prompt) - 1 + i])
            base_nll = -float(np.log(max(float(p[token]), 1e-12)))
            hit = memory.lookup(ctx) if memory is not None else None
            if hit is not None:
                key, entry = hit
                q = memory_dist(entry, p.shape[0])
                fused_nll = [
                    -float(
                        np.log(
                            max(
                                float(fuse_probs(p, q, lam, "arithmetic")[token]),
                                1e-12,
                            )
                        )
                    )
                    for lam in lambdas
                ]
                top_prob = float(p.max())
                for gi, gate in enumerate(gates):
                    if not gate.allows(key, entry, top_prob):
                        continue
                    fired[gi] += 1
                    row = gains[gi]
                    for li in range(len(lambdas)):
                        row[li] += base_nll - fused_nll[li]
            ctx.append(token)

    rows: list[dict] = []
    best: dict | None = None
    for gi, gate in enumerate(gates):
        coverage = fired[gi] / positions if positions else 0.0
        for li, lam in enumerate(lambdas):
            dev_gain = gains[gi][li] / positions if positions else 0.0
            row = {
                "gate": gate.as_dict(),
                "lam": lam,
                "dev_gain": round(dev_gain, 6),
                "coverage": round(coverage, 4),
            }
            rows.append(row)
            if coverage >= float(min_coverage) and (
                best is None or row["dev_gain"] > best["dev_gain"]
            ):
                best = dict(row)

    return {"rows": rows, "best": best, "n_positions": positions}
