"""增量 9:预算约束下的内容选择。dev 统计 → 键排序 → 子表。

选择层证据(增量 4 的 C1)说明速度最优集与质量最优集几乎不相交;
本模块把每个键在 dev 上的使用情况量化成三个统计量,供五种预算策略排序。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from everyngram.fuse import fuse_probs, memory_dist, softmax

STRATEGIES = ("freq", "query", "quality", "speed", "hybrid")


def score_entry_usage(model, memory, cases, *, lam: float, gate=None) -> dict:
    """在 dev 用例上累计每个键的查询、质量增益与接受次数。

    - queries:该键被 lookup 命中的次数;
    - quality:Σ(基础 NLL − 融合 NLL),只计 gate 放行的位置;
    - accepted:草稿首 token 等于模型 argmax 的次数(贪婪验证的接受事件)。
    """
    per_key: dict[tuple[int, ...], dict] = {}
    positions = 0
    hits = 0
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
            hit = memory.lookup(ctx)
            if hit is not None:
                key, entry = hit
                hits += 1
                row = per_key.setdefault(
                    key, {"queries": 0, "quality": 0.0, "accepted": 0}
                )
                row["queries"] += 1
                if entry.continuation and int(entry.continuation[0]) == int(p.argmax()):
                    row["accepted"] += 1
                if gate is None or gate.allows(key, entry, float(p.max())):
                    q = memory_dist(entry, p.shape[0])
                    fused = fuse_probs(p, q, lam, "arithmetic")
                    base_nll = -float(np.log(max(float(p[token]), 1e-12)))
                    fused_nll = -float(np.log(max(float(fused[token]), 1e-12)))
                    row["quality"] += base_nll - fused_nll
            ctx.append(token)
    return {"per_key": per_key, "positions": positions, "hits": hits}


def _freq_order(memory) -> list[tuple[int, ...]]:
    """与 Memory.build 的预算截断同序:按 (阶数, 频次) 降序,稳定排序。"""
    ranked = sorted(
        memory.entries.items(),
        key=lambda kv: (len(kv[0]), kv[1].count),
        reverse=True,
    )
    return [key for key, _ in ranked]


def _stat_order(memory, usage, field: str) -> list[tuple[int, ...]]:
    """按统计量降序排全部键,同分按 freq 序(保证确定性)。"""
    per_key = usage.get("per_key", {})
    index = {key: i for i, key in enumerate(_freq_order(memory))}
    return sorted(
        memory.entries,
        key=lambda key: (
            -float(per_key.get(key, {}).get(field, 0.0)),
            index[key],
        ),
    )


def select_keys(
    memory, usage, *, strategy: str, budget: int
) -> tuple[tuple[int, ...], ...]:
    """按策略选出 min(budget, 全表) 个键;同分一律按 freq 序打破。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略 {strategy}: 必须是 {STRATEGIES} 之一")
    budget = int(budget)
    if budget < 0:
        raise ValueError(f"budget 必须 >= 0: {budget}")
    size = min(budget, len(memory.entries))
    if size == 0:
        return ()

    freq = _freq_order(memory)
    if strategy == "freq":
        return tuple(freq[:size])

    query = _stat_order(memory, usage, "queries")
    quality = _stat_order(memory, usage, "quality")
    speed = _stat_order(memory, usage, "accepted")

    if strategy == "query":
        return tuple(query[:size])
    if strategy == "quality":
        return tuple(quality[:size])
    if strategy == "speed":
        return tuple(speed[:size])

    half = max(1, size // 2)
    picked = list(quality[:half])
    seen = set(picked)
    for key in speed[:half]:
        if key not in seen:
            picked.append(key)
            seen.add(key)
    for key in query + freq:
        if len(picked) >= size:
            break
        if key not in seen:
            picked.append(key)
            seen.add(key)
    return tuple(picked[:size])
