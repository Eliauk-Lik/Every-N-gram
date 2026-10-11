"""增量 13:任务层效应的协议化(复制型载体,dev 选 λ + 留出评估 + OOD 代价)。

用法:
    python -u scripts/bench_inc13.py --config configs/inc13.yaml
    python -u scripts/bench_inc13.py --config configs/inc13.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import plot_task_layer, run_task_layer_experiment  # noqa: E402
from everyngram.hf import (  # noqa: E402
    HFModel,
    corpus_token_ids,
    load_tokenizer,
    tokenizer_fingerprint,
)
from everyngram.memory import Memory  # noqa: E402
from everyngram.tasks import (  # noqa: E402
    build_corpus_ids,
    collect_code_files,
    collect_table_prompts,
)

DEFAULT_CONFIG = Path("configs/inc13.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true", help="压规模跑链路")
    return parser.parse_args()


def verdict(result: dict) -> dict:
    paired = result["test_paired"]
    selected = result["selected"]
    test_row = next(
        row
        for row in result["rows"]
        if row["split"] == "test" and row["config"] == selected["config"]
    )
    ood_max = max(
        (float(item["ratio_vs_plain"]) for item in result["ood"]), default=None
    )
    return {
        "T1_task_layer": {
            "mean": paired["mean"],
            "t": paired["t"],
            "ci": [paired["ci_low"], paired["ci_high"]],
            "passes": bool(
                paired["mean"] > 0
                and (paired["t"] or 0) >= 2
                and (paired["ci_low"] is not None and paired["ci_low"] > 0)
            ),
        },
        "T2_dose_response": {
            "dev_spearman": result["dev_spearman"],
            "passes": bool((result["dev_spearman"] or 0) >= 0.8),
        },
        "T3_changed": {
            "mean_changed": test_row["mean_changed"],
            "passes": bool(test_row["mean_changed"] >= 0.2),
        },
        "T4_descriptive": {
            "exact": test_row["exact"],
            "n_test": len(test_row["per_scenario"]),
            "ood_max_ratio": ood_max,
        },
    }


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke = bool(args.smoke)

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(str(cfg["corpus"]))
    if not corpus_dir.exists():
        raise FileNotFoundError(f"语料目录不存在: {corpus_dir.resolve()}")
    table_sample = int(cfg["table_sample"])
    dev_count = int(cfg["dev_scenarios"])
    scenario_offset = int(cfg.get("scenario_offset", 0))
    test_count = int(cfg.get("test_scenarios", 0))
    horizon = int(cfg["horizon"])
    lambdas = tuple(float(lam) for lam in cfg["lambdas"])
    if smoke:
        table_sample, dev_count = 32, 8
        scenario_offset, test_count = 16, 8
        horizon = 8
        lambdas = (0.2, 0.4)

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
    prompts = collect_table_prompts(
        sorted(corpus_dir.rglob("*.txt")),
        tokenizer,
        n=table_sample,
        context_len=int(cfg["context_len"]),
        tail_chars=int(cfg["tail_chars"]),
        horizon=horizon,
    )
    window = prompts[
        scenario_offset : scenario_offset
        + dev_count
        + (test_count if test_count else len(prompts))
    ]
    scenarios = [
        {
            "name": f"copy-{scenario_offset + index:02d}",
            "prompt_ids": item["prompt_ids"],
            "reference_ids": item["reference_ids"],
            "origin": item.get("origin"),
        }
        for index, item in enumerate(window)
    ]
    dev_scenarios = scenarios[:dev_count]
    test_scenarios = (
        scenarios[dev_count : dev_count + test_count]
        if test_count
        else scenarios[dev_count:]
    )
    if not dev_scenarios or not test_scenarios:
        raise ValueError(
            f"场景不足:需要超过 {dev_count} 个(实得 {len(scenarios)})"
        )

    ood_cfg = cfg["ood"]
    ood_file_count = int(ood_cfg["files"])
    if smoke:
        ood_file_count = 2
    ood_files = collect_code_files(
        [str(ood_cfg["root"])],
        max_files=ood_file_count,
        seed=0,
        exclude_substrings=tuple(ood_cfg.get("exclude", ())),
    )
    ood_ids = build_corpus_ids(ood_files, tokenizer)[
        : int(ood_cfg.get("max_tokens", 4000))
    ]
    print(
        f"[记忆] {memory.stats()} | 场景 dev={len(dev_scenarios)} test={len(test_scenarios)}"
        f" | OOD {len(ood_files)} 文件/{len(ood_ids)} token",
        flush=True,
    )

    result = run_task_layer_experiment(
        model,
        memory,
        dev_scenarios,
        test_scenarios,
        lambdas=lambdas,
        horizon=horizon,
        ood_ids=ood_ids,
        bootstrap=int(cfg["bootstrap"]),
    )
    for row in result["rows"]:
        print(
            f"  {row['split']:<4} {row['config']:<12}"
            f" prefix={row['mean_prefix']:.4f}±{row['se_prefix']:.4f}"
            f" changed={row['mean_changed']:.3f} exact={row['exact']}",
            flush=True,
        )
    for item in result["ood"]:
        print(
            f"  OOD {item['config']}: ppl={item['perplexity']}"
            f" (x{item['ratio_vs_plain']})",
            flush=True,
        )
    result["verdict"] = verdict(result)
    print(f"[判据] {result['verdict']}", flush=True)

    report = {
        "config": {
            "model": cfg["model"],
            "dtype": cfg.get("dtype", "float32"),
            "corpus": str(corpus_dir),
            "table_sample": table_sample,
            "dev_scenarios": len(dev_scenarios),
            "scenario_offset": scenario_offset,
            "test_scenarios": len(test_scenarios),
            "horizon": horizon,
            "lambdas": list(lambdas),
            "ood_files": len(ood_files),
            "ood_tokens": len(ood_ids),
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
    plot_task_layer(result, png_path)
    print(f"[输出] {json_path} 与 {png_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
