"""增量 10:复制型载体的速度轴协议化读数(与自由生成对照)。

用法:
    python -u scripts/bench_inc10.py --config configs/inc10.yaml
    python -u scripts/bench_inc10.py --config configs/inc10.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import (  # noqa: E402
    plot_copy_speed,
    run_copy_speed_experiment,
)
from everyngram.hf import (  # noqa: E402
    HFModel,
    corpus_token_ids,
    load_tokenizer,
    tokenizer_fingerprint,
)
from everyngram.memory import Memory  # noqa: E402
from everyngram.tasks import collect_table_prompts  # noqa: E402

DEFAULT_CONFIG = Path("configs/inc10.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true", help="压规模跑链路")
    return parser.parse_args()


def verdict(summary: dict) -> dict:
    """按预注册判据(C10.1 到 C10.3)给出机器可读的判定。"""
    out: dict = {}
    best_copy = None
    for gamma, block in sorted(summary.get("copy", {}).items()):
        passed = (
            block["ci_low"] is not None
            and block["ci_low"] > 1.0
            and block["mean_accept_length"] > 2.08
        )
        out[f"copy_gamma{gamma}_passes_C10.1"] = passed
        if passed:
            if best_copy is None or block["mean"] > best_copy[1]:
                best_copy = (gamma, block["mean"])
    out["C10.1_any_gamma"] = best_copy is not None
    copy_blocks = summary.get("copy", {})
    free_blocks = summary.get("free", {})
    if copy_blocks and free_blocks:
        gamma = max(copy_blocks, key=lambda g: copy_blocks[g]["mean_accept_length"])
        copy_accept = copy_blocks[gamma]["mean_accept_length"]
        free_accept = free_blocks.get(gamma, {}).get("mean_accept_length")
        best_free = max(free_blocks, key=lambda g: free_blocks[g]["mean"])
        free_speedup = free_blocks[best_free]["mean"]
        out["C10.2"] = {
            "gamma": gamma,
            "copy_accept_length": copy_accept,
            "free_accept_length": free_accept,
            "free_best_speedup": free_speedup,
            "passes": bool(
                free_accept is not None
                and copy_accept > float(free_accept) + 1.0
                and free_speedup < 1.0
            ),
        }
    all_identical = all(
        block["all_identical"] for block in summary.get("copy", {}).values()
    ) and all(block["all_identical"] for block in summary.get("free", {}).values())
    out["C10.3_all_identical"] = bool(all_identical)
    return out


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke = bool(args.smoke)

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(str(cfg["corpus"]))
    if not corpus_dir.exists():
        raise FileNotFoundError(f"语料目录不存在: {corpus_dir.resolve()}")
    files = sorted(corpus_dir.rglob("*.txt"))
    copy_scenarios = int(cfg["copy_scenarios"])
    gammas = tuple(int(g) for g in cfg["gammas"])
    max_new_tokens = int(cfg["max_new_tokens"])
    repeats = int(cfg["repeats"])
    free_prompts = [str(p) for p in cfg["free_prompts"]]
    if smoke:
        copy_scenarios = min(copy_scenarios, 2)
        free_prompts = free_prompts[:2]
        gammas = (4,)
        max_new_tokens = 16
        repeats = 1

    tokenizer = load_tokenizer(cfg["model"])
    fingerprint = tokenizer_fingerprint(tokenizer)
    model = HFModel(cfg["model"], dtype=str(cfg.get("dtype", "float32")))
    memory_cfg = cfg["memory"]
    memory = Memory.build(
        corpus_token_ids(tokenizer, corpus_dir),
        fingerprint=fingerprint,
        orders=tuple(memory_cfg["orders"]),
        budget=int(memory_cfg["budget"]),
        continuation_len=int(memory_cfg["continuation_len"]),
    )
    print(f"[记忆] {memory.stats()}", flush=True)

    copy_prompts = collect_table_prompts(
        files,
        tokenizer,
        n=copy_scenarios,
        context_len=int(cfg["context_len"]),
        tail_chars=int(cfg["tail_chars"]),
    )
    scenarios = []
    for index, item in enumerate(copy_prompts):
        scenarios.append(
            {
                "name": f"copy-{index}",
                "arm": "copy",
                "prompt_ids": item["prompt_ids"],
                "origin": item["origin"],
            }
        )
    for index, prompt in enumerate(free_prompts):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        scenarios.append(
            {
                "name": f"free-{index}",
                "arm": "free",
                "prompt_ids": tuple(ids),
                "origin": "instruction",
            }
        )
    print(
        f"[场景] copy {len(copy_prompts)} 个,free {len(free_prompts)} 个;"
        f"gammas={gammas} max_new_tokens={max_new_tokens} repeats={repeats}",
        flush=True,
    )
    if not copy_prompts:
        raise ValueError("没有抽到表格场景:检查语料里是否存在表格行")

    result = run_copy_speed_experiment(
        model,
        memory,
        scenarios,
        gammas=gammas,
        max_new_tokens=max_new_tokens,
        repeats=repeats,
        bootstrap=int(cfg["bootstrap"]),
    )
    for arm, blocks in result["summary"].items():
        for gamma, block in sorted(blocks.items()):
            print(
                f"  {arm:<4} gamma={gamma}: speedup={block['mean']:+.3f}"
                f"±{block['se']:.3f} CI[{block['ci_low']}, {block['ci_high']}]"
                f" accept_len={block['mean_accept_length']:.3f}"
                f" hit={block['mean_hit_rate']:.3f}"
                f" >1: {block['frac_above_one']:.2f}"
                f" identical={block['all_identical']}",
                flush=True,
            )
    result["verdict"] = verdict(result["summary"])
    print(f"[判据] {result['verdict']}", flush=True)

    report = {
        "config": {
            "model": cfg["model"],
            "dtype": cfg.get("dtype", "float32"),
            "corpus": str(corpus_dir),
            "copy_scenarios": len(copy_prompts),
            "free_prompts": len(free_prompts),
            "gammas": list(gammas),
            "max_new_tokens": max_new_tokens,
            "repeats": repeats,
            "smoke": smoke,
        },
        **result,
    }
    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    png_path = out.with_suffix(".png")
    plot_copy_speed(result, png_path)
    print(f"[输出] {json_path} 与 {png_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
