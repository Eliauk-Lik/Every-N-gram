"""增量 6 端到端:多折 × 多语料的门控实验,带文件级聚类稳健统计。

用法:
    python scripts/bench_inc6.py --config configs/inc6.yaml
    python scripts/bench_inc6.py --config configs/inc6.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import (  # noqa: E402
    make_folds,
    plot_fold_experiment,
    run_fold_experiment,
)
from everyngram.fuse import Gate  # noqa: E402
from everyngram.hf import HFModel, load_tokenizer, tokenizer_fingerprint  # noqa: E402
from everyngram.tasks import collect_code_files  # noqa: E402

DEFAULT_CONFIG = Path("configs/inc6.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true", help="压规模跑链路")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke = bool(args.smoke)

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    folds = int(cfg["folds"])
    max_files = int(cfg["max_files"])
    dev_cases = int(cfg["dev_cases"])
    test_cases = int(cfg["test_cases"])
    repeats = int(cfg["repeats"])
    lambdas = tuple(float(lam) for lam in cfg["lambdas"])
    gates = tuple(Gate(**raw) for raw in cfg["gates"])
    if smoke:
        folds = 3
        max_files = 9
        dev_cases = min(dev_cases, 3)
        test_cases = min(test_cases, 3)
        repeats = 1
        gates = gates[:1]
        lambdas = lambdas[:1]

    tokenizer = load_tokenizer(cfg["model"])
    fingerprint = tokenizer_fingerprint(tokenizer)
    model = HFModel(cfg["model"], dtype=str(cfg.get("dtype", "float32")))
    print(f"[模型] {cfg['model']} dtype={cfg.get('dtype', 'float32')}")

    report: dict = {
        "config": {
            "model": cfg["model"],
            "dtype": cfg.get("dtype", "float32"),
            "folds": folds,
            "gates": [gate.as_dict() for gate in gates],
            "lambdas": list(lambdas),
            "min_coverage": float(cfg["min_coverage"]),
            "smoke": smoke,
        },
        "corpora": {},
    }
    for entry in cfg["corpora"]:
        name = str(entry["name"])
        root = str(entry["root"])
        if not Path(root).exists():
            raise FileNotFoundError(f"语料目录不存在: {Path(root).resolve()}")
        files = collect_code_files(
            [root],
            max_files=max_files,
            exclude_substrings=tuple(entry.get("exclude", ())),
        )
        print(f"[语料 {name}] 文件 {len(files)}(根 {root})")
        if len(files) < 3 * folds:
            print(f"[语料 {name}] 文件数不足 {3 * folds},跳过并把原因写进报告")
            report["corpora"][name] = {
                "files": len(files),
                "skipped": f"文件数不足 {3 * folds}",
            }
            continue
        result = run_fold_experiment(
            model,
            tokenizer,
            folds=make_folds(files, k=folds, seed=0),
            fingerprint=fingerprint,
            gates=gates,
            lambdas=lambdas,
            dev_cases=dev_cases,
            test_cases=test_cases,
            corpus_orders=tuple(cfg["orders"]),
            min_coverage=float(cfg["min_coverage"]),
            gamma=int(cfg["gamma"]),
            prefix_tokens=int(cfg["prefix_tokens"]),
            timing_tokens=int(cfg["timing_tokens"]),
            repeats=repeats,
        )
        report["corpora"][name] = {"files": len(files), **result}
        pooled = result["pooled"].get("gated")
        if pooled:
            print(
                f"[语料 {name}] gated 折间均值 {pooled['mean_across_folds']:+.4f} "
                f"± {pooled['se_across_folds']:.4f}({pooled['per_fold_improved']}/"
                f"{pooled['n_folds']} 折为正);逐折 {pooled['per_fold_delta']}"
            )
        else:
            print(f"[语料 {name}] 没有任何一折选出可用配置")

    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    png_path = out.with_suffix(".png")
    plot_fold_experiment(report, png_path)
    print(f"[输出] {json_path}、{png_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
