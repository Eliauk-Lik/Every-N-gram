"""增量 8:记忆预算的剂量-反应曲线 + 记忆抽样方差。

每个 run 用同一 pool/seed 切分,记忆表规模按嵌套子集从小到大,
dev/test 完全固定,因此曲线上的点两两可比(配对)。

用法:
    python scripts/bench_inc8.py --config configs/inc8.yaml
    python scripts/bench_inc8.py --config configs/inc8.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import run_power_experiment  # noqa: E402
from everyngram.fuse import Gate  # noqa: E402
from everyngram.hf import HFModel, load_tokenizer, tokenizer_fingerprint  # noqa: E402

DEFAULT_CONFIG = Path("configs/inc8.yaml")


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
    runs = list(cfg["runs"])
    if smoke:
        full_pool = int(cfg["pool_files"])
        pool_files, dev_files, test_files = 12, 6, 12
        dev_cases_per_file, test_cases_per_file = 2, 2
        gates, lambdas = gates[:1], lambdas[:1]
        runs = runs[:2]
        runs = [
            {
                **spec,
                "memory_files": max(
                    2, round(int(spec["memory_files"]) * pool_files / full_pool)
                ),
            }
            for spec in runs
        ]

    tokenizer = load_tokenizer(cfg["model"])
    fingerprint = tokenizer_fingerprint(tokenizer)
    model = HFModel(cfg["model"], dtype=str(cfg.get("dtype", "float32")))
    print(f"[模型] {cfg['model']} dtype={cfg.get('dtype', 'float32')};共 {len(runs)} 个 run")

    rows: list[dict] = []
    for spec in runs:
        name = str(spec["name"])
        entry = cfg["corpora"][str(spec["corpus"])]
        root = Path(str(entry["root"]))
        if not root.exists():
            raise FileNotFoundError(f"语料目录不存在: {root.resolve()}")
        block = run_power_experiment(
            model,
            tokenizer,
            corpus_root=str(root),
            exclude=tuple(entry.get("exclude", ())),
            fingerprint=fingerprint,
            gates=gates,
            lambdas=lambdas,
            memory_files=int(spec["memory_files"]),
            pool_files=pool_files,
            memory_offset=int(spec.get("memory_offset", 0)),
            dev_files=dev_files,
            test_files=test_files,
            dev_cases_per_file=dev_cases_per_file,
            test_cases_per_file=test_cases_per_file,
            corpus_orders=tuple(cfg["orders"]),
            min_coverage=float(cfg["min_coverage"]),
            em_tokens=int(cfg["em_tokens"]),
            bootstrap=int(cfg["bootstrap"]),
        )
        selected = block.get("selected")
        gated = (block.get("test") or {}).get("gated")
        row = {
            "name": name,
            "corpus": str(spec["corpus"]),
            "memory_files": int(spec["memory_files"]),
            "memory_offset": int(spec.get("memory_offset", 0)),
            "entries": block["counts"]["entries"],
            "corpus_tokens": block["counts"]["corpus_tokens"],
            "selected": selected,
            "baseline_nll": (block.get("test") or {}).get("baseline", {}).get("nll"),
            "gated_mean": None if gated is None else gated["mean"],
            "gated_se": None if gated is None else gated["se"],
            "gated_ci_low": None if gated is None else gated["ci_low"],
            "gated_ci_high": None if gated is None else gated["ci_high"],
            "gated_nll": None if gated is None else gated["nll"],
            "improved_files": None if gated is None else gated["improved_groups"],
            "n_test_files": None if gated is None else gated["n_test_files"],
            "em1_gated": None if gated is None else gated.get("em1"),
            "em1_baseline": (block.get("test") or {}).get("baseline", {}).get("em1"),
        }
        rows.append(row)
        if selected and gated:
            print(
                f"[{name}] 表 {row['entries']} 条({row['memory_files']} 文件)"
                f" 选中 λ={selected['lam']} gate={selected['gate']} 覆盖={selected['coverage']}"
                f" | G={gated['mean']:+.4f}±{gated['se']:.4f}"
                f" CI[{gated['ci_low']:+.4f},{gated['ci_high']:+.4f}]"
                f" {gated['improved_groups']}/{gated['n_test_files']}"
                f" em1={gated.get('em1')} vs {row['em1_baseline']}",
                flush=True,
            )
        else:
            print(f"[{name}] 未选出配置或没有可用测试用例", flush=True)

    report = {
        "config": {
            "model": cfg["model"],
            "dtype": cfg.get("dtype", "float32"),
            "pool_files": pool_files,
            "dev_files": dev_files,
            "test_files": test_files,
            "smoke": smoke,
        },
        "rows": rows,
    }
    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[输出] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
