"""R7:可插拔评测任务。O1 定档 A(代码续写),B/C 未实现。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from everyngram.fuse import fused_next_probs, memory_dist, softmax


@dataclass(frozen=True)
class EvalCase:
    """一个评测用例:给定前缀,要求续写参考片段。

    origin 记录来源文件,供文件级聚类稳健统计使用(增量 6)。
    """

    prompt_ids: tuple[int, ...]
    reference_ids: tuple[int, ...]
    origin: str = ""


class EvalTask(Protocol):
    """评测任务的最小接口。"""

    name: str
    cases: tuple[EvalCase, ...]

    def score(self, output_ids: Sequence[int], case: EvalCase) -> float:
        """0 到 1 的得分,越高越好。"""
        ...


def collect_code_files(
    roots: Sequence[str | Path],
    *,
    max_files: int = 40,
    max_bytes: int = 200_000,
    min_bytes: int = 2_000,
    seed: int = 0,
    exclude_substrings: Sequence[str] = (),
) -> list[Path]:
    """收集本地 .py 文件,按大小过滤、按路径排除后固定种子抽样。"""
    files: list[Path] = []
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            if any(part in str(path) for part in exclude_substrings):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if min_bytes <= size <= max_bytes:
                files.append(path)
    random.Random(seed).shuffle(files)
    return files[:max_files]


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _is_separator_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    core = stripped.replace("|", "").strip()
    return bool(core) and all(ch in "-: " for ch in core)


def collect_table_docs(
    roots: Sequence[str | Path],
    *,
    max_files: int = 60,
    min_bytes: int = 2_000,
    max_bytes: int = 200_000,
    patterns: Sequence[str] = ("*.md", "METADATA"),
    min_table_lines: int = 2,
    exclude_substrings: Sequence[str] = (),
) -> list[Path]:
    """收集含 markdown 表格行的本地文档,按表格行数降序、路径升序打破并列。"""
    scored: list[tuple[int, str, Path]] = []
    for root in roots:
        base = Path(root)
        for pattern in patterns:
            for path in sorted(base.rglob(pattern)):
                if any(
                    part in str(path) for part in tuple(exclude_substrings)
                ):
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if not (int(min_bytes) <= size <= int(max_bytes)):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                table_lines = sum(
                    1
                    for line in text.splitlines()
                    if line.lstrip().startswith("|")
                )
                if table_lines < int(min_table_lines):
                    continue
                scored.append((table_lines, str(path), path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [path for _count, _name, path in scored[: int(max_files)]]


def collect_table_prompts(
    files: Sequence[str | Path],
    tokenizer,
    *,
    n: int = 16,
    context_len: int = 128,
    tail_chars: int = 2000,
    horizon: int = 32,
    seed: int = 0,
) -> tuple[dict, ...]:
    """抽取表格续写场景:切点取表格块内从第二行起的行首。

    提示词是切点之前 tail_chars 个字符再截到 context_len 个 token;
    reference_ids 是切点之后的 horizon 个 token(任务层读数用)。
    只依赖文本结构(行首 `|`),同一输入与种子产出同一批场景。
    """
    candidates: list[tuple[Path, int]] = []
    for path in files:
        target = Path(path)
        text = target.read_text(encoding="utf-8", errors="ignore")
        run = 0
        offset = 0
        for line in text.splitlines(keepends=True):
            if _is_table_line(line):
                run += 1
                if run >= 2 and not _is_separator_line(line):
                    candidates.append((target, offset))
            else:
                run = 0
            offset += len(line)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    prompts: list[dict] = []
    cache: dict[Path, str] = {}
    for target, cut in candidates:
        if len(prompts) >= int(n):
            break
        if target not in cache:
            cache[target] = target.read_text(encoding="utf-8", errors="ignore")
        window = cache[target][max(0, cut - int(tail_chars)) : cut]
        ids = [int(t) for t in tokenizer.encode(window, add_special_tokens=False)]
        if int(context_len) > 0:
            ids = ids[-int(context_len) :]
        if not ids:
            continue
        tail_ids = [
            int(t)
            for t in tokenizer.encode(
                cache[target][cut:], add_special_tokens=False
            )
        ]
        prompts.append(
            {
                "prompt_ids": tuple(ids),
                "reference_ids": tuple(tail_ids[: int(horizon)]),
                "origin": str(target),
                "char_offset": int(cut),
            }
        )
    return tuple(prompts)


def split_files(
    files: Sequence[Path], *, holdout_ratio: float = 0.2
) -> tuple[list[Path], list[Path]]:
    """按顺序切分:前段建表,后段做评测用例,保证语料与用例不重叠。"""
    items = list(files)
    if len(items) < 2:
        raise ValueError("至少需要两个文件才能切分")
    cut = max(1, int(len(items) * (1.0 - holdout_ratio)))
    cut = min(cut, len(items) - 1)
    return items[:cut], items[cut:]


def build_corpus_ids(files: Sequence[Path], tokenizer) -> list[int]:
    """把 .py 文件拼成分词后的 token 序列。"""
    ids: list[int] = []
    for path in files:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
    return ids


@dataclass(frozen=True)
class CompletionTask:
    """代码续写任务:exact match 打分。"""

    name: str
    cases: tuple[EvalCase, ...]

    def score(self, output_ids: Sequence[int], case: EvalCase) -> float:
        horizon = len(case.reference_ids)
        return 1.0 if tuple(output_ids[:horizon]) == case.reference_ids else 0.0


def build_completion_task(
    files: Sequence[Path],
    tokenizer,
    *,
    name: str = "code_completion",
    n_cases: int = 32,
    context_len: int = 128,
    horizon: int = 8,
    seed: int = 0,
) -> CompletionTask:
    """从 held-out 文件里截取前缀与参考,构造 exact match 用例。"""
    rng = random.Random(seed)
    paths = list(files)
    per_file = max(1, -(-int(n_cases) // max(len(paths), 1)))
    cases: list[EvalCase] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        ids = tokenizer.encode(text, add_special_tokens=False)
        span = len(ids) - context_len - horizon
        if span < 1:
            continue
        for start in rng.sample(range(span + 1), min(per_file, span + 1)):
            cases.append(
                EvalCase(
                    prompt_ids=tuple(ids[start : start + context_len]),
                    reference_ids=tuple(
                        ids[start + context_len : start + context_len + horizon]
                    ),
                    origin=str(path),
                )
            )
            if len(cases) >= n_cases:
                break
        if len(cases) >= n_cases:
            break
    return CompletionTask(name=name, cases=tuple(cases))


def next_token_accuracy(model, cases: Sequence[EvalCase], fusion=None) -> float:
    """域内指标:argmax 命中参考首 token 的比例(teacher forcing)。

    fusion 给出时 argmax 取自融合分布 p_λ,未命中位置自动退回 p_θ。
    """
    if not cases:
        return 0.0
    hits = 0
    for case in cases:
        logits, _ = model.forward(list(case.prompt_ids))
        probs = fused_next_probs(logits[-1], case.prompt_ids, fusion)
        if int(np.argmax(probs)) == case.reference_ids[0]:
            hits += 1
    return hits / len(cases)


def perplexity(
    model, token_ids: Sequence[int], fusion=None, *, window: int = 256
) -> float:
    """域外指标:按窗口切块的困惑度,NLL 用融合分布算。"""
    ids = [int(t) for t in token_ids]
    if len(ids) < 2:
        raise ValueError("至少需要两个 token")

    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(ids) - 1, window):
        chunk = ids[start : start + window + 1]
        if len(chunk) < 2:
            continue
        logits, _ = model.forward(chunk)
        for i in range(len(chunk) - 1):
            probs = fused_next_probs(logits[i], chunk[: i + 1], fusion)
            prob = max(float(probs[chunk[i + 1]]), 1e-12)
            total_nll -= float(np.log(prob))
            total_tokens += 1

    return float(np.exp(total_nll / max(total_tokens, 1)))


def tv_distance(model, cases: Sequence[EvalCase], memory) -> dict:
    """分布距离读数:TV(p_θ, q),只在记忆命中的位置计入均值。

    未命中位置的 q 是空分布,TV(p, 0) 恒为 0.5,计入均值会污染读数,
    因此单独记为 hit_rate。
    """
    if not cases:
        return {"tv_mean": None, "hit_rate": 0.0, "n_hits": 0, "n_cases": 0}
    total = 0.0
    hits = 0
    for case in cases:
        logits, _ = model.forward(list(case.prompt_ids))
        p = softmax(logits[-1])
        hit = memory.lookup(list(case.prompt_ids)) if memory is not None else None
        if hit is None:
            continue
        q = memory_dist(hit[1], p.shape[0])
        if q.sum() <= 0:
            continue
        total += 0.5 * float(np.abs(p - q).sum())
        hits += 1
    return {
        "tv_mean": (total / hits) if hits else None,
        "hit_rate": hits / len(cases),
        "n_hits": hits,
        "n_cases": len(cases),
    }


def _case_nll(model, case: EvalCase, fusion=None) -> float:
    """单个用例的 teacher forcing 平均 NLL;参考为空时返回 0。"""
    prompt = list(case.prompt_ids)
    reference = list(case.reference_ids)
    if not reference:
        return 0.0
    logits, _ = model.forward(prompt + reference[:-1])
    total = 0.0
    ctx = list(prompt)
    for i, token in enumerate(reference):
        probs = fused_next_probs(logits[len(prompt) - 1 + i], ctx, fusion)
        total -= float(np.log(max(float(probs[token]), 1e-12)))
        ctx.append(token)
    return total / len(reference)


def next_token_nll(model, cases: Sequence[EvalCase], fusion=None) -> float:
    """teacher forcing 的负对数似然,按 token 平均;对分布移动敏感。"""
    if not cases:
        return 0.0
    return float(np.mean([_case_nll(model, case, fusion) for case in cases]))


def paired_nll_delta(model, cases: Sequence[EvalCase], fusion=None) -> dict:
    """逐用例的配对 NLL 增益(基线 − 融合),按用例聚合均值与标准误。

    同一用例内部的多个位置相关,按位置聚合会低估方差,因此配对单位是用例。
    """
    deltas = [
        _case_nll(model, case, None) - _case_nll(model, case, fusion)
        for case in cases
    ]
    n = len(deltas)
    if n == 0:
        return {"n": 0, "mean": 0.0, "se": 0.0, "improved": 0, "deltas": []}
    mean = float(np.mean(deltas))
    se = float(np.std(deltas, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "improved": int(sum(1 for delta in deltas if delta > 0)),
        "deltas": [float(delta) for delta in deltas],
    }


def completion_exact_match(
    model, cases: Sequence[EvalCase], fusion=None, *, max_new_tokens: int = 8
) -> float:
    """任务层读数:贪心生成的前 max_new_tokens 个 token 与参考逐位相同的比例。"""
    from everyngram.decode import greedy_generate

    if not cases:
        return 0.0
    hits = 0
    for case in cases:
        output, _ = greedy_generate(
            model,
            list(case.prompt_ids),
            max_new_tokens=int(max_new_tokens),
            fusion=fusion,
            use_cache=True,
        )
        reference = list(case.reference_ids)[: int(max_new_tokens)]
        if output[: len(reference)] == reference:
            hits += 1
    return hits / len(cases)


def completion_prefix_score(
    model, cases: Sequence[EvalCase], fusion=None, *, max_new_tokens: int = 8
) -> float:
    """任务层读数:生成与参考的最长公共前缀长度,按 max_new_tokens 归一后取平均。

    相比精确匹配,这个口径在基线上也取非零值,因此有分辨力(审计指出
    8-token 精确匹配在基线上就是 0,无法区分好坏)。
    """
    from everyngram.decode import greedy_generate

    if not cases:
        return 0.0
    horizon = max(int(max_new_tokens), 1)
    scores = []
    for case in cases:
        output, _ = greedy_generate(
            model,
            list(case.prompt_ids),
            max_new_tokens=horizon,
            fusion=fusion,
            use_cache=True,
        )
        reference = list(case.reference_ids)[:horizon]
        matched = 0
        for produced, expected in zip(output, reference):
            if produced != expected:
                break
            matched += 1
        scores.append(matched / horizon)
    return float(np.mean(scores))


def clustered_delta_stats(pairs, *, bootstrap: int = 0, seed: int = 0) -> dict:
    """文件级聚类稳健的配对统计:先在文件内平均,再在文件间算均值与标准误。

    pairs 是 (文件标识, 用例差值) 的序列。同一文件内的用例高度相关,
    直接在用例上算 SE 会低估不确定性(审计指出 test 只有 2 个文件时的偏差)。

   """
    if int(bootstrap) < 0:
        raise ValueError(f"bootstrap 必须 >= 0: {bootstrap}")
    grouped: dict[str, list[float]] = {}
    for group, delta in pairs:
        grouped.setdefault(str(group), []).append(float(delta))
    if not grouped:
        return {
            "n_cases": 0,
            "n_groups": 0,
            "mean": 0.0,
            "se": 0.0,
            "improved_groups": 0,
            "ci_low": None,
            "ci_high": None,
            "bootstrap": int(bootstrap),
        }
    group_means = [float(np.mean(values)) for values in grouped.values()]
    n_groups = len(group_means)
    mean = float(np.mean(group_means))
    se = float(np.std(group_means, ddof=1) / np.sqrt(n_groups)) if n_groups > 1 else 0.0
    ci_low = ci_high = None
    if int(bootstrap) > 0:
        rng = np.random.default_rng(seed)
        means = np.asarray(group_means)
        draws = rng.integers(0, n_groups, size=(int(bootstrap), n_groups))
        resampled = means[draws].mean(axis=1)
        ci_low = float(np.percentile(resampled, 2.5))
        ci_high = float(np.percentile(resampled, 97.5))
    return {
        "n_cases": int(sum(len(values) for values in grouped.values())),
        "n_groups": n_groups,
        "mean": mean,
        "se": se,
        "improved_groups": int(sum(1 for value in group_means if value > 0)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap": int(bootstrap),
    }
