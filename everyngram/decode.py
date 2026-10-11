"""S4: 贪婪解码与投机解码。只依赖 numpy 与一个返回 logits 的协议。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from everyngram.fuse import FusionSpec, fused_next_probs


class TokenModel(Protocol):
    """解码循环需要的最小模型接口。"""

    def next_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        """返回每个位置之后的 next-token logits,形状 (len(token_ids), vocab)。"""
        ...


@runtime_checkable
class CachedModel(Protocol):
    """带缓存的模型接口。缓存对象由实现自定义,解码循环只透传。"""

    def forward(
        self, token_ids: Sequence[int], cache=None
    ) -> tuple[np.ndarray, object]:
        """喂入新 token,返回这些新位置的 logits 与更新后的缓存。"""
        ...

    def crop(self, cache, keep: int) -> object:
        """把缓存裁剪到前 keep 个位置,用于拒绝回滚。"""
        ...


@dataclass
class DecodeStats:
    """一次解码过程的读数。"""

    steps: int = 0
    produced: int = 0
    drafted: int = 0
    accepted: int = 0
    lookups: int = 0
    hits: int = 0
    probe_steps: int = 0
    probe_agreed: int = 0
    switched: bool = False

    @property
    def accept_length(self) -> float:
        """每步平均产出的 token 数。纯贪婪等于 1。"""
        return self.produced / self.steps if self.steps else 0.0

    @property
    def hit_rate(self) -> float:
        """查询命中记忆表的位置占比。"""
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def accept_rate(self) -> float:
        """草稿 token 中被接受的比例。"""
        return self.accepted / self.drafted if self.drafted else 0.0

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "produced": self.produced,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "lookups": self.lookups,
            "hits": self.hits,
            "probe_steps": self.probe_steps,
            "probe_agreed": self.probe_agreed,
            "switched": self.switched,
            "accept_length": round(self.accept_length, 4),
            "hit_rate": round(self.hit_rate, 4),
            "accept_rate": round(self.accept_rate, 4),
        }


def greedy_generate(
    model: TokenModel,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    memory=None,
    gamma: int = 0,
    use_cache: bool = False,
    fusion: FusionSpec | None = None,
) -> tuple[list[int], DecodeStats]:
    """贪婪解码。memory 为 None 或 gamma <= 0 时走普通路径。

    use_cache=True 时走带 KV 缓存的通路,要求 model 实现 CachedModel。
    fusion 给出融合配置时,取 next token 的依据是融合分布 p_λ。
    """
    ctx = [int(t) for t in prompt_ids]
    if not ctx:
        raise ValueError("prompt_ids 不能为空")
    if max_new_tokens <= 0:
        return [], DecodeStats()
    if memory is not None and gamma > 0:
        if use_cache:
            return _speculative_greedy_cached(
                model, ctx, max_new_tokens, memory, gamma, fusion
            )
        if fusion is not None:
            raise NotImplementedError("无缓存投机通路的融合留给后续增量")
        return _speculative_greedy(model, ctx, max_new_tokens, memory, gamma)
    if use_cache:
        return _plain_greedy_cached(model, ctx, max_new_tokens, fusion)
    return _plain_greedy(model, ctx, max_new_tokens, fusion)


def adaptive_greedy_generate(
    model: CachedModel,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    memory,
    gamma: int,
    probe_steps: int = 8,
    min_agree: float = 0.5,
    fusion: FusionSpec | None = None,
) -> tuple[list[int], DecodeStats]:
    """在线探针的贪婪解码:先普通走 probe_steps 步,再决定剩余走哪条通路。

    探针阶段每一步只做普通解码本就要做的那次前向,额外成本是查表与比较;
    一致率(草稿首 token == 目标 argmax)达到 min_agree 时剩余 token 走投机,
    否则继续普通通路。两条通路都是无损的,输出与纯普通解码逐位相同。
    """
    if memory is None:
        raise ValueError("adaptive_greedy_generate 需要 memory")
    if not 0.0 <= float(min_agree) <= 1.0:
        raise ValueError(f"min_agree 必须在 [0, 1] 内: {min_agree}")
    ctx = [int(t) for t in prompt_ids]
    if not ctx:
        raise ValueError("prompt_ids 不能为空")
    stats = DecodeStats()
    if int(max_new_tokens) <= 0:
        return [], stats

    logits, cache = model.forward(ctx)
    next_logits = logits[-1]
    probe = max(0, min(int(probe_steps), int(max_new_tokens)))
    agreed = 0
    out: list[int] = []
    for _ in range(probe):
        probs = fused_next_probs(next_logits, ctx, fusion)
        nxt = int(np.argmax(probs))
        stats.lookups += 1
        hit = memory.lookup(ctx)
        if hit is not None:
            stats.hits += 1
            entry = hit[1]
            if entry.continuation and int(entry.continuation[0]) == nxt:
                agreed += 1
        ctx.append(nxt)
        out.append(nxt)
        stats.steps += 1
        stats.produced += 1
        if stats.produced < int(max_new_tokens):
            logits, cache = model.forward([nxt], cache)
            next_logits = logits[-1]
    stats.probe_steps = probe
    stats.probe_agreed = agreed

    remaining = int(max_new_tokens) - probe
    if remaining <= 0:
        return out, stats
    rate = agreed / probe if probe > 0 else 0.0
    if rate >= float(min_agree):
        stats.switched = True
        extra, _, _ = _speculative_loop_cached(
            model,
            ctx,
            next_logits,
            cache,
            remaining,
            memory,
            gamma,
            fusion,
            stats,
        )
    else:
        extra, _, _ = _plain_loop_cached(
            model, ctx, next_logits, cache, remaining, fusion, stats
        )
    return out + extra, stats


def _plain_greedy(
    model: TokenModel, ctx: list[int], max_new_tokens: int, fusion=None
) -> tuple[list[int], DecodeStats]:
    stats = DecodeStats()
    out: list[int] = []
    for _ in range(max_new_tokens):
        logits = model.next_logits(ctx)
        probs = fused_next_probs(logits[-1], ctx, fusion)
        nxt = int(np.argmax(probs))
        ctx.append(nxt)
        out.append(nxt)
        stats.steps += 1
        stats.produced += 1
    return out, stats


def _plain_greedy_cached(
    model: CachedModel, ctx: list[int], max_new_tokens: int, fusion=None
) -> tuple[list[int], DecodeStats]:
    """缓存版贪婪基线:prefill 一次,之后每步只前向一个新 token。"""
    stats = DecodeStats()
    out: list[int] = []
    logits, cache = model.forward(ctx)
    out, _, _ = _plain_loop_cached(
        model, ctx, logits[-1], cache, int(max_new_tokens), fusion, stats
    )
    return out, stats


def _plain_loop_cached(
    model: CachedModel,
    ctx: list[int],
    next_logits,
    cache,
    remaining: int,
    fusion,
    stats: DecodeStats,
) -> tuple[list[int], object, object]:
    """普通解码循环:从已预填的 next_logits/cache 继续,产出 remaining 个 token。"""
    out: list[int] = []
    while len(out) < int(remaining):
        probs = fused_next_probs(next_logits, ctx, fusion)
        nxt = int(np.argmax(probs))
        ctx.append(nxt)
        out.append(nxt)
        stats.steps += 1
        stats.produced += 1
        if len(out) < int(remaining):
            logits, cache = model.forward([nxt], cache)
            next_logits = logits[-1]
    return out, cache, next_logits


def _speculative_greedy_cached(
    model: CachedModel,
    ctx: list[int],
    max_new_tokens: int,
    memory,
    gamma: int,
    fusion: FusionSpec | None = None,
) -> tuple[list[int], DecodeStats]:
    """缓存版贪婪投机解码。与无缓存版本输出与读数逐项相同。

    fusion 给出时验证目标换成 p_λ,fusion 为 None 时退回 p_θ。
    """
    stats = DecodeStats()
    logits, cache = model.forward(ctx)
    out, _, _ = _speculative_loop_cached(
        model,
        ctx,
        logits[-1],
        cache,
        int(max_new_tokens),
        memory,
        gamma,
        fusion,
        stats,
    )
    return out, stats


def _speculative_loop_cached(
    model: CachedModel,
    ctx: list[int],
    next_logits,
    cache,
    remaining: int,
    memory,
    gamma: int,
    fusion: FusionSpec | None,
    stats: DecodeStats,
) -> tuple[list[int], object, object]:
    """投机解码循环:从已预填的 next_logits/cache 继续,产出 remaining 个 token。

    每轮验证后,修正 token 与下一轮草稿合并成一次前向
    (`forward([correction] + draft_next)`):第 0 行 logits 验证 draft_next[0],
    第 i 行验证 draft_next[i]。因此每轮只需一次前向(空草稿回退除外),
    而旧实现是"验证 + bonus"两次;输出与逐位无损性不变。
    """
    out: list[int] = []
    pending: dict | None = None

    def fetch_draft() -> list[int]:
        stats.lookups += 1
        draft = list(memory.draft(ctx, gamma))
        if draft:
            stats.hits += 1
            stats.drafted += len(draft)
        return draft

    def build_preds(draft: list[int], head, tail) -> list:
        if fusion is None:
            return [head] + list(tail)
        # preds[i] 是"上下文 + 前 i 个草稿 token"之后的靶分布:
        # 第 0 个用预填/修正位置的 logits,第 i 个用草稿第 i-1 个位置的 logits。
        preds = [fused_next_probs(head, ctx, fusion)]
        for i, row in enumerate(tail):
            preds.append(fused_next_probs(row, list(ctx) + draft[: i + 1], fusion))
        return preds

    while len(out) < int(remaining):
        left = int(remaining) - len(out)
        if pending is not None and not pending["draft"]:
            # 上一轮已确认"无草稿":直接普通步,不重复查表
            pending = None
            nxt = int(np.argmax(next_logits))
            ctx.append(nxt)
            out.append(nxt)
            stats.steps += 1
            stats.produced += 1
            if len(out) < int(remaining):
                logits, cache = model.forward([nxt], cache)
                next_logits = logits[-1]
            continue

        if pending is not None:
            draft = pending["draft"]
            preds = build_preds(draft, pending["head"], pending["tail"])
            pending = None
        else:
            draft = fetch_draft()
            if not draft:
                nxt = int(np.argmax(next_logits))
                ctx.append(nxt)
                out.append(nxt)
                stats.steps += 1
                stats.produced += 1
                if len(out) < int(remaining):
                    logits, cache = model.forward([nxt], cache)
                    next_logits = logits[-1]
                continue
            draft_logits, cache = model.forward(draft, cache)
            preds = build_preds(draft, next_logits, list(draft_logits))

        verified = 0
        for i, token in enumerate(draft):
            if int(np.argmax(preds[i])) == token:
                verified += 1
            else:
                break

        kept = min(verified, left)
        emit = draft[:kept]
        if len(emit) < left:
            emit = emit + [int(np.argmax(preds[kept]))]

        cache = model.crop(cache, len(ctx) + kept)
        ctx.extend(emit)
        out.extend(emit)
        stats.steps += 1
        stats.accepted += kept
        stats.produced += len(emit)

        if len(out) < int(remaining):
            bonus = emit[-1]
            next_draft = fetch_draft()
            if next_draft:
                logits, cache = model.forward([bonus] + next_draft, cache)
                pending = {
                    "draft": next_draft,
                    "head": logits[0],
                    "tail": list(logits[1:]),
                }
            else:
                logits, cache = model.forward([bonus], cache)
                next_logits = logits[-1]
                pending = {"draft": [], "head": None, "tail": None}

    return out, cache, next_logits


def _sample_index(rng, probs) -> int:
    """按离散分布采样一个下标。"""
    return int(rng.choice(len(probs), p=np.asarray(probs, dtype=np.float64)))


def _residual(p, q) -> np.ndarray:
    """残差分布:归一化的 max(0, p - q)。全零时回退到 p。"""
    r = np.maximum(
        np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64), 0.0
    )
    total = r.sum()
    if total <= 0:
        return np.asarray(p, dtype=np.float64)
    return r / total


def _speculative_draw(p, q, token, rng) -> tuple[int, bool]:
    """给定从 q 采样的 token,按 min(1, p/q) 接受;拒绝时按残差重采样。"""
    q_t = float(q[token])
    ratio = float(p[token]) / q_t if q_t > 0 else 1.0
    if rng.random() < min(1.0, ratio):
        return token, True
    return _sample_index(rng, _residual(p, q)), False


def sample_generate(
    model: CachedModel,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    memory=None,
    gamma: int = 0,
    temperature: float = 1.0,
    seed: int = 0,
    fusion: FusionSpec | None = None,
) -> tuple[list[int], DecodeStats]:
    """采样版投机解码。草稿按 q 采样,按 min(1, p_λ/q) 接受,拒绝时残差重采样。

    fusion 给出时靶分布换成融合分布 p_λ,草稿仍按记忆分布 q 采样。
    """
    rng = np.random.default_rng(seed)
    ctx = [int(t) for t in prompt_ids]
    if not ctx:
        raise ValueError("prompt_ids 不能为空")
    stats = DecodeStats()
    out: list[int] = []
    if max_new_tokens <= 0:
        return out, stats

    logits, cache = model.forward(ctx)

    while len(out) < max_new_tokens:
        remaining = max_new_tokens - len(out)
        p_next = fused_next_probs(logits[-1], ctx, fusion, temperature)
        stats.lookups += 1

        draft: list[int] = []
        qs: list[np.ndarray] = []
        look_ctx = list(ctx)
        for _ in range(max(0, int(gamma))):
            hit = memory.lookup(look_ctx) if memory is not None else None
            if hit is None:
                break
            _, entry = hit
            q = np.zeros(p_next.shape[0], dtype=np.float64)
            for token, count in entry.next_top:
                if 0 <= int(token) < q.shape[0]:
                    q[int(token)] = float(count)
            if q.sum() <= 0:
                break
            q = q / q.sum()
            token = _sample_index(rng, q)
            draft.append(token)
            qs.append(q)
            look_ctx = look_ctx + [token]

        if not draft:
            nxt = _sample_index(rng, p_next)
            ctx.append(nxt)
            out.append(nxt)
            stats.steps += 1
            stats.produced += 1
            if len(out) < max_new_tokens:
                logits, cache = model.forward([nxt], cache)
            continue

        stats.hits += 1
        stats.drafted += len(draft)
        draft_logits, cache = model.forward(draft, cache)
        preds = [p_next]
        for i, row in enumerate(draft_logits):
            preds.append(
                fused_next_probs(row, list(ctx) + draft[:i], fusion, temperature)
            )

        accepted = 0
        rejection: int | None = None
        for i, token in enumerate(draft):
            sampled, ok = _speculative_draw(preds[i], qs[i], token, rng)
            if ok:
                accepted += 1
            else:
                rejection = sampled
                break

        emit = draft[:accepted]
        if len(emit) < remaining:
            if rejection is not None:
                emit = emit + [rejection]
            elif accepted == len(draft):
                emit = emit + [_sample_index(rng, preds[len(draft)])]
        if len(emit) > remaining:
            emit = emit[:remaining]

        cache = model.crop(cache, len(ctx) + min(accepted, len(emit)))
        ctx.extend(emit)
        out.extend(emit)
        stats.steps += 1
        stats.accepted += min(accepted, len(emit))
        stats.produced += len(emit)

        if len(out) < max_new_tokens:
            logits, cache = model.forward([emit[-1]], cache)

    return out, stats


def _speculative_greedy(
    model: TokenModel,
    ctx: list[int],
    max_new_tokens: int,
    memory,
    gamma: int,
) -> tuple[list[int], DecodeStats]:
    """贪婪投机解码。与 _plain_greedy 的输出必须逐 token 相同。"""
    stats = DecodeStats()
    out: list[int] = []

    while len(out) < max_new_tokens:
        remaining = max_new_tokens - len(out)
        stats.lookups += 1
        draft = list(memory.draft(ctx, gamma))

        if not draft:
            logits = model.next_logits(ctx)
            nxt = int(np.argmax(logits[-1]))
            ctx.append(nxt)
            out.append(nxt)
            stats.steps += 1
            stats.produced += 1
            continue

        stats.hits += 1
        stats.drafted += len(draft)

        # 无状态前向:必须把完整上下文一起送进模型。
        # 只送 ctx[-1] 加草稿的做法,只在模型自带 KV 缓存时成立。
        window = ctx + draft
        logits = model.next_logits(window)
        offset = len(ctx) - 1

        verified = 0
        for i, token in enumerate(draft):
            if int(np.argmax(logits[offset + i])) == token:
                verified += 1
            else:
                break

        kept = min(verified, remaining)
        emit = draft[:kept]
        if len(emit) < remaining:
            emit = emit + [int(np.argmax(logits[offset + verified]))]

        ctx.extend(emit)
        out.extend(emit)
        stats.steps += 1
        stats.accepted += kept
        stats.produced += len(emit)

    return out, stats
