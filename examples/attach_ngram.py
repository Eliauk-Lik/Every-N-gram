"""把 n-gram 条件记忆挂到任意模型上的最小示例:建表 → 探针 → 双出口 → 在线门控。

用法(默认用本机 Qwen3-0.6B 与项目文档语料,零网络):
    python examples/attach_ngram.py \
        --corpus data/project-docs \
        --prompt "Every N-gram 是一个工具链"

换自己的语料:--corpus 指向任意文本目录(递归读 .txt);
换自己的模型:--model 指向本地 HF 目录(需要能给出 next-token logits)。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.decode import (  # noqa: E402
    adaptive_greedy_generate,
    greedy_generate,
)
from everyngram.fuse import FusionSpec, softmax  # noqa: E402
from everyngram.hf import (  # noqa: E402
    HFModel,
    corpus_token_ids,
    load_tokenizer,
    tokenizer_fingerprint,
)
from everyngram.memory import Memory  # noqa: E402

DEFAULT_PROBES = [
    "| 出口 A(速度) | 表作为草稿",
    "| 一张表 | 从离线语料构建",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/models/Qwen3-0.6B")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--corpus", default="data/project-docs")
    parser.add_argument("--memory-in", default="")
    parser.add_argument("--memory-out", default="")
    parser.add_argument("--orders", default="1,2,3")
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--continuation-len", type=int, default=16)
    parser.add_argument("--prompt", default="| 出口 A(速度) | 表作为草稿")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gamma", type=int, default=8)
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--probe-horizon", type=int, default=8)
    return parser.parse_args()


def load_or_build_memory(args, tokenizer) -> tuple[Memory, float]:
    fingerprint = tokenizer_fingerprint(tokenizer)
    if args.memory_in:
        memory = Memory.load(args.memory_in)
        if memory.fingerprint != fingerprint:
            raise SystemExit(
                "记忆表与当前模型的分词器不一致(指纹不同),拒绝加载:"
                f"表={memory.fingerprint} 模型={fingerprint}"
            )
        return memory, 0.0
    ids = corpus_token_ids(tokenizer, args.corpus)
    if not ids:
        raise SystemExit(f"语料为空: {args.corpus}(需要 .txt 文件)")
    start = time.perf_counter()
    memory = Memory.build(
        ids,
        fingerprint=fingerprint,
        orders=tuple(int(x) for x in args.orders.split(",") if x.strip()),
        budget=args.budget,
        continuation_len=args.continuation_len,
    )
    seconds = time.perf_counter() - start
    if args.memory_out:
        memory.save(args.memory_out)
    return memory, seconds


def probe(memory: Memory, model: HFModel, tokenizer, prompts, horizon: int) -> dict:
    """两个便宜探针:命中率与 a_model(草稿首 token 与模型 argmax 的逐步一致率)。

    a_model 高 → 速度出口值得试;a_target 要拿参考续写才能测(见 bench_inc15_align.py),
    这里只演示模型侧。"""
    agreed = hits = steps = 0
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            continue
        logits, cache = model.forward(ids)
        ctx = list(ids)
        for step in range(int(horizon)):
            hit = memory.lookup(ctx)
            nxt = int(softmax(logits[-1]).argmax())
            if hit is not None:
                hits += 1
                entry = hit[1]
                if entry.continuation and int(entry.continuation[0]) == nxt:
                    agreed += 1
            steps += 1
            ctx.append(nxt)
            if step < int(horizon) - 1:
                logits, cache = model.forward([nxt], cache)
    return {
        "a_model": agreed / max(steps, 1),
        "hit_rate": hits / max(steps, 1),
        "steps": steps,
    }


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.model)
    memory, build_seconds = load_or_build_memory(args, tokenizer)
    print(f"[表] {memory.stats()}  build={build_seconds:.2f}s", flush=True)

    model = HFModel(args.model, dtype=args.dtype)
    stats = probe(memory, model, tokenizer, DEFAULT_PROBES, args.probe_horizon)
    print(f"[探针] {stats}", flush=True)

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise SystemExit("prompt 为空")
    # 预热:第一次前向包含 CUDA 初始化,不计时也不计入任何结论
    greedy_generate(model, prompt_ids, max_new_tokens=1)

    plain, _ = greedy_generate(
        model, prompt_ids, max_new_tokens=args.max_new_tokens, use_cache=True
    )
    spec, spec_stats = greedy_generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        memory=memory,
        gamma=args.gamma,
        use_cache=True,
    )
    fused, _ = greedy_generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        fusion=FusionSpec(lam=args.lam, memory=memory),
    )
    adaptive, adaptive_stats = adaptive_greedy_generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        memory=memory,
        gamma=args.gamma,
        probe_steps=8,
        min_agree=0.5,
    )

    changed = sum(1 for a, b in zip(plain, fused) if a != b)
    print("[出口] 速度(spec):", spec_stats.as_dict(), flush=True)
    print(
        "[出口] 速度(adaptive):",
        adaptive_stats.as_dict(),
        f"switched={adaptive_stats.switched}",
        flush=True,
    )
    print(
        f"[核对] plain==spec: {plain == spec};plain==adaptive: {plain == adaptive};"
        f"融合改动 token 数: {changed}/{len(plain)}",
        flush=True,
    )
    print("[文本] plain   :", tokenizer.decode(plain), flush=True)
    print("[文本] fused   :", tokenizer.decode(fused), flush=True)
    print(
        "[提示] 这只是链路演示:计时要用重复中位数(scripts/bench_inc*.py),"
        "收益是否为正取决于你语料与模型的对齐度(先看探针,见 README 方法一节)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
