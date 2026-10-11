"""S1: n-gram 条件记忆表。零机器学习依赖。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

FORMAT_VERSION = 1


@dataclass(frozen=True)
class Entry:
    """一个 n-gram 键对应的记忆条目。"""

    count: int
    next_top: tuple[tuple[int, int], ...]
    continuation: tuple[int, ...]


class Memory:
    """从 token 序列构建的 n-gram 表。"""

    def __init__(
        self,
        *,
        fingerprint: str,
        orders: Sequence[int],
        budget: int,
        continuation_len: int,
        next_top_k: int,
        entries: dict[tuple[int, ...], Entry],
        dropped: int = 0,
    ) -> None:
        self.fingerprint = fingerprint
        self.orders = tuple(sorted(int(o) for o in orders))
        self.budget = int(budget)
        self.continuation_len = int(continuation_len)
        self.next_top_k = int(next_top_k)
        self.entries = dict(entries)
        self.dropped = int(dropped)

    @classmethod
    def build(
        cls,
        token_ids: Sequence[int],
        *,
        fingerprint: str,
        orders: Sequence[int] = (1, 2, 3),
        budget: int = 200_000,
        continuation_len: int = 16,
        next_top_k: int = 8,
    ) -> "Memory":
        ids = [int(t) for t in token_ids]
        budget = int(budget)
        if budget < 0:
            raise ValueError(f"budget 必须 >= 0: {budget}")

        order_list = tuple(sorted({int(o) for o in orders}))
        if not order_list or order_list[0] < 1:
            raise ValueError("orders 必须是非空的正整数集合")
        max_order = order_list[-1]
        if len(ids) <= max_order:
            raise ValueError(
                f"序列长度 {len(ids)} 不足以构建最高 {max_order} 阶的键"
            )

        counts: dict[tuple[int, ...], Counter] = {}
        first_continuation: dict[tuple[int, ...], tuple[int, ...]] = {}

        for k in order_list:
            for i in range(len(ids) - k):
                key = tuple(ids[i : i + k])
                nxt = ids[i + k]
                counts.setdefault(key, Counter())[nxt] += 1
                if key not in first_continuation:
                    first_continuation[key] = tuple(
                        ids[i + k : i + k + continuation_len]
                    )

        entries = {
            key: Entry(
                count=int(sum(counter.values())),
                next_top=tuple(
                    (int(t), int(c)) for t, c in counter.most_common(next_top_k)
                ),
                continuation=first_continuation[key],
            )
            for key, counter in counts.items()
        }

        dropped = 0
        if len(entries) > budget:
            ranked = sorted(
                entries.items(),
                key=lambda kv: (len(kv[0]), kv[1].count),
                reverse=True,
            )
            kept = dict(ranked[:budget])
            dropped = len(entries) - len(kept)
            entries = kept

        return cls(
            fingerprint=fingerprint,
            orders=order_list,
            budget=budget,
            continuation_len=continuation_len,
            next_top_k=next_top_k,
            entries=entries,
            dropped=dropped,
        )

    def lookup(
        self, context: Sequence[int]
    ) -> tuple[tuple[int, ...], Entry] | None:
        """从最高阶往低阶查找,返回 (键, 条目);全部未命中返回 None。"""
        ctx = tuple(int(t) for t in context)
        for k in reversed(self.orders):
            if len(ctx) < k:
                continue
            key = ctx[-k:]
            entry = self.entries.get(key)
            if entry is not None:
                return key, entry
        return None

    def draft(self, context: Sequence[int], gamma: int) -> tuple[int, ...]:
        """按最长匹配取续写片段,截断到 gamma 个 token;未命中返回空元组。"""
        gamma = int(gamma)
        if gamma <= 0:
            return ()
        hit = self.lookup(context)
        if hit is None:
            return ()
        _, entry = hit
        return tuple(entry.continuation[:gamma])

    def subset(
        self,
        keys: Sequence[tuple[int, ...] | Sequence[int]],
        *,
        budget: int | None = None,
    ) -> "Memory":
        """返回只保留给定键的子表;未知键报错,丢弃量如实记账。

        条目对象与父表共享(不复制),键的迭代顺序沿用父表的插入顺序,
        保证同一父表 + 同一键集合产出的子表是确定性的。
        """
        wanted = {tuple(int(t) for t in key) for key in keys}
        unknown = wanted - set(self.entries)
        if unknown:
            sample = sorted(unknown)[:3]
            raise KeyError(
                f"subset 收到 {len(unknown)} 个不存在的键,例如 {sample}"
            )
        kept = {key: entry for key, entry in self.entries.items() if key in wanted}
        return Memory(
            fingerprint=self.fingerprint,
            orders=self.orders,
            budget=int(budget) if budget is not None else len(kept),
            continuation_len=self.continuation_len,
            next_top_k=self.next_top_k,
            entries=kept,
            dropped=self.dropped + (len(self.entries) - len(kept)),
        )

    def stats(self) -> dict:
        """返回条目数、被丢弃数与各阶条目分布。"""
        by_order = Counter(len(key) for key in self.entries)
        return {
            "entries": len(self.entries),
            "dropped": self.dropped,
            "by_order": {str(k): int(by_order.get(k, 0)) for k in self.orders},
        }

    def save(self, path: str | Path) -> None:
        """把记忆表写成 .ngmem 文件。键按排序输出,保证字节可复现。"""
        payload = {
            "format_version": FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "orders": list(self.orders),
            "budget": self.budget,
            "continuation_len": self.continuation_len,
            "next_top_k": self.next_top_k,
            "dropped": self.dropped,
            "entries": [
                {
                    "key": list(key),
                    "count": entry.count,
                    "next_top": [list(pair) for pair in entry.next_top],
                    "continuation": list(entry.continuation),
                }
                for key, entry in sorted(self.entries.items())
            ],
        }
        text = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Memory":
        """从 .ngmem 文件读回记忆表。"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"不支持的格式版本: {version}")
        entries = {
            tuple(int(t) for t in item["key"]): Entry(
                count=int(item["count"]),
                next_top=tuple(
                    (int(t), int(c)) for t, c in item["next_top"]
                ),
                continuation=tuple(int(t) for t in item["continuation"]),
            )
            for item in payload["entries"]
        }
        return cls(
            fingerprint=payload["fingerprint"],
            orders=payload["orders"],
            budget=int(payload["budget"]),
            continuation_len=int(payload["continuation_len"]),
            next_top_k=int(payload["next_top_k"]),
            entries=entries,
            dropped=int(payload["dropped"]),
        )
