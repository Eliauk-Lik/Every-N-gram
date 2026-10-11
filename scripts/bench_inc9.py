"""增量 9:预算约束下的内容选择(同预算换内容,两语料)。

用法:
    python -u scripts/bench_inc9.py --config configs/inc9.yaml
    python -u scripts/bench_inc9.py --config configs/inc9.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import (  # noqa: E402
    plot_budget_selection,
    run_budget_selection_experiment,
)
from everyngram.fuse import Gate  # noqa: E402
from everyngram.hf import HFModel, load_tokenizer, tokenizer_fingerprint  # noqa: E402

DEFAULT_CONFIG = Path("configs/inc9.yaml")


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
    pool_files = int(cfg["pool_files"])
    dev_files, test_files = int(cfg["dev_files"]), int(cfg["test_files"])
    dev_cases_per_file = int(cfg["dev_cases_per_file"])
    test_cases_per_file = int(cfg["test_cases_per_file"])
    gates = tuple(Gate(**raw) for raw in cfg["gates"])
    lambdas = tuple(float(lam) for lam in cfg["lambdas"])
    budgets = tuple(int(b) for b in cfg["budgets"])
    strategies = tuple(str(s) for s in cfg["strategies"])
    runs = list(cfg["runs"])
    min_coverage = float(cfg["min_coverage"])
    if smoke:
        pool_files, dev_files, test_files = 12, 6, 12
        dev_cases_per_file, test_cases_per_file = 2, 2
        budgets = tuple(budgets[:2])
        min_coverage = 0.0
        runs = runs[:1]

    tokenizer = load_tokenizer(cfg["model"])
    fingerprint = tokenizer_fingerprint(tokenizer)
    model = HFModel(cfg["model"], dtype=str(cfg.get("dtype", "float32")))
    print(
        f"[模型] {cfg['model']} dtype={cfg.get('dtype', 'float32')};"
        f"共 {len(runs)} 个语料",
        flush=True,
    )

    blocks: list[dict] = []
    for spec in runs:
        name = str(spec["name"])
        entry = cfg["corpora"][str(spec["corpus"])]
        root = Path(str(entry["root"]))
        if not root.exists():
            raise FileNotFoundError(f"语料目录不存在: {root.resolve()}")
        block = run_budget_selection_experiment(
            model,
            tokenizer,
            corpus_root=str(root),
            exclude=tuple(entry.get("exclude", ())),
            fingerprint=fingerprint,
            gates=gates,
            lambdas=lambdas,
            pool_files=pool_files,
            dev_files=dev_files,
            test_files=test_files,
            dev_cases_per_file=dev_cases_per_file,
            test_cases_per_file=test_cases_per_file,
            corpus_orders=tuple(cfg["orders"]),
            min_coverage=min_coverage,
            budgets=budgets,
            strategies=strategies,
            bootstrap=int(cfg["bootstrap"]),
        )
        block["name"] = name
        block["corpus"] = str(spec["corpus"])
        blocks.append(block)
        selected = block.get("selected")
        usage = block.get("dev_usage", {})
        print(
            f"[{name}] 全表 {block['counts']['entries']} 条;"
            f"dev 命中 {usage.get('hits')}/{usage.get('positions')} 位置,"
            f"{usage.get('keys_used')} 个不同键"
            + (
                f";选中 λ={selected['lam']} gate={selected['gate']}"
                if selected
                else ";未选出配置"
            ),
            flush=True,
        )
        for row in block.get("rows", []):
            if row["strategy"] == "full":
                continue
            print(
                f"  B={row['entries']:>6} {row['strategy']:<8}"
                f" G={row['mean']:+.4f}±{row['se']:.4f}"
                f" hit={row['hit_rate']:.4f}"
                f" accept_cov={row['accept_coverage']:.4f}"
                f" accept_rate={row['accept_rate']}",
                flush=True,
            )
        if block.get("overlap"):
            print(
                "  overlap(quality,speed): "
                + ", ".join(
                    f"B={item['entries']}:{item['overlap']:.3f}"
                    for item in block["overlap"]
                ),
                flush=True,
            )
        png = out.with_name(f"{out.name}_{name}.png")
        plot_budget_selection({"rows": block.get("rows", [])}, png)
        print(f"[图] {png}", flush=True)

    report = {
        "config": {
            "model": cfg["model"],
            "dtype": cfg.get("dtype", "float32"),
            "pool_files": pool_files,
            "dev_files": dev_files,
            "test_files": test_files,
            "dev_cases_per_file": dev_cases_per_file,
            "test_cases_per_file": test_cases_per_file,
            "budgets": list(budgets),
            "strategies": list(strategies),
            "smoke": smoke,
        },
        "blocks": blocks,
    }
    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[输出] {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
