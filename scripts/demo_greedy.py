"""增量 1 端到端演示:建表 -> 两种贪婪解码 -> 打印读数。

用法(必须先设镜像,否则 huggingface.co 不通):
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python scripts/demo_greedy.py --corpus data/project-docs --prompt "Every N-gram 是一个工具链"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.decode import greedy_generate  # noqa: E402
from everyngram.hf import (  # noqa: E402
    HFModel,
    corpus_token_ids,
    load_tokenizer,
    tokenizer_fingerprint,
)
from everyngram.memory import Memory  # noqa: E402

LIMITATION = (
    "增量 1 使用无状态前向,复杂度 O(n^2),延迟数字不可用于生产结论;"
    "语料为项目自身文档,命中率不代表真实领域数据。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--orders", default="1,2,3")
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--continuation-len", type=int, default=16)
    parser.add_argument("--memory-out", default="")
    parser.add_argument("--memory-in", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orders = tuple(int(x) for x in args.orders.split(",") if x.strip())

    tokenizer = load_tokenizer(args.model)
    fingerprint = tokenizer_fingerprint(tokenizer)

    start = time.perf_counter()
    corpus_ids = corpus_token_ids(tokenizer, args.corpus)
    corpus_seconds = time.perf_counter() - start

    if args.memory_in:
        memory = Memory.load(args.memory_in)
        if memory.fingerprint != fingerprint:
            print(
                "错误:记忆表的 tokenizer 指纹与当前模型不一致,拒绝加载。"
                f"表={memory.fingerprint} 模型={fingerprint}",
                file=sys.stderr,
            )
            return 2
        build_seconds = 0.0
    else:
        start = time.perf_counter()
        memory = Memory.build(
            corpus_ids,
            fingerprint=fingerprint,
            orders=orders,
            budget=args.budget,
            continuation_len=args.continuation_len,
        )
        build_seconds = time.perf_counter() - start
        if args.memory_out:
            memory.save(args.memory_out)

    model = HFModel(args.model)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)

    # 预热:第一次前向包含 CUDA 初始化与算子编译,必须排除在计时之外,
    # 否则先跑的那一路会把预热成本算进去,产生虚假的加速。
    greedy_generate(model, prompt_ids, max_new_tokens=1)

    start = time.perf_counter()
    plain, plain_stats = greedy_generate(
        model, prompt_ids, max_new_tokens=args.max_new_tokens
    )
    plain_seconds = time.perf_counter() - start

    start = time.perf_counter()
    spec, spec_stats = greedy_generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        memory=memory,
        gamma=args.gamma,
    )
    spec_seconds = time.perf_counter() - start

    identical = plain == spec
    summary = {
        "model": args.model,
        "corpus_tokens": len(corpus_ids),
        "orders": list(orders),
        "gamma": args.gamma,
        "memory": memory.stats(),
        "timing_s": {
            "corpus": round(corpus_seconds, 3),
            "build": round(build_seconds, 3),
            "plain": round(plain_seconds, 3),
            "speculative": round(spec_seconds, 3),
        },
        "outputs_identical": identical,
        "plain_stats": plain_stats.as_dict(),
        "spec_stats": spec_stats.as_dict(),
        "text_plain": tokenizer.decode(plain),
        "text_spec": tokenizer.decode(spec),
        "limitation": LIMITATION,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not identical:
        print("错误:投机解码输出与基线不一致,停止。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
