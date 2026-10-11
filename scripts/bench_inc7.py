"""增量 7 端到端:大样本判定(一次冻结 + 240 个 test 文件),按预先写下的判据出结论。

用法:
    python scripts/bench_inc7.py --config configs/inc7.yaml
    python scripts/bench_inc7.py --config configs/inc7.yaml --smoke
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

DEFAULT_CONFIG = Path("configs/inc7.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true", help="压规模跑链路")
    return parser.parse_args()


def judge(block: dict) -> dict:
    """按 spec 3.3 的三条判据给结论;不做事后修饰。"""
    gated = block.get("test", {}).get("gated")
    if not gated or gated.get("ci_low") is None:
        return {"conclusion": "undetermined", "reason": "没有可用区间(未选出配置或无测试文件)"}
    low, high = gated["ci_low"], gated["ci_high"]
    em_ok = gated.get("em1") is None or gated.get("em1") >= block["test"]["baseline"]["em1"]
    if low > 0 and em_ok:
        return {"conclusion": "positive", "reason": f"区间下界 {low:+.4f} > 0 且守门指标不劣"}
    if high < 0:
        return {"conclusion": "negative", "reason": f"区间上界 {high:+.4f} < 0"}
    need = ""
    if gated["mean"] > 0 and gated["se"] > 0:
        # 达到 t=3 还需要多少文件:SE 与 1/sqrt(n) 成比例
        factor = (3.0 * gated["se"] / gated["mean"]) ** 2
        need = f";按当前离散度,达到 t=3 约需 {factor:.1f} 倍文件数"
    return {
        "conclusion": "undetermined",
        "reason": f"区间 [{low:+.4f}, {high:+.4f}] 跨 0{need}",
    }


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke = bool(args.smoke)

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    memory_files = int(cfg["memory_files"])
    dev_files = int(cfg["dev_files"])
    test_files = int(cfg["test_files"])
    dev_cases_per_file = int(cfg["dev_cases_per_file"])
    test_cases_per_file = int(cfg["test_cases_per_file"])
    gates = tuple(Gate(**raw) for raw in cfg["gates"])
    lambdas = tuple(float(lam) for lam in cfg["lambdas"])
    if smoke:
        memory_files, dev_files, test_files = 8, 8, 24
        dev_cases_per_file, test_cases_per_file = 2, 2
        gates, lambdas = gates[:1], lambdas[:1]

    tokenizer = load_tokenizer(cfg["model"])
    fingerprint = tokenizer_fingerprint(tokenizer)
    model = HFModel(cfg["model"], dtype=str(cfg.get("dtype", "float32")))
    print(f"[模型] {cfg['model']} dtype={cfg.get('dtype', 'float32')}")

    report: dict = {
        "config": {
            "model": cfg["model"],
            "dtype": cfg.get("dtype", "float32"),
            "memory_files": memory_files,
            "dev_files": dev_files,
            "test_files": test_files,
            "gates": [gate.as_dict() for gate in gates],
            "lambdas": list(lambdas),
            "min_coverage": float(cfg["min_coverage"]),
            "smoke": smoke,
        },
        "corpora": {},
    }
    for entry in cfg["corpora"]:
        name = str(entry["name"])
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
            memory_files=memory_files,
            dev_files=dev_files,
            test_files=test_files,
            dev_cases_per_file=dev_cases_per_file,
            test_cases_per_file=test_cases_per_file,
            corpus_orders=tuple(cfg["orders"]),
            min_coverage=float(cfg["min_coverage"]),
            em_tokens=int(cfg["em_tokens"]),
            bootstrap=int(cfg["bootstrap"]),
        )
        verdict = judge(block)
        block["verdict"] = verdict
        report["corpora"][name] = block
        selected = block.get("selected")
        print(f"[语料 {name}] 文件 memory={block['counts']['memory']} "
              f"dev={block['counts']['dev']} test={block['counts']['test']};"
              f"条目 {block['counts']['entries']};dev 用例 {block['dev']['n_cases']}")
        if selected:
            print(f"          选中 gate={selected['gate']} λ={selected['lam']} "
                  f"覆盖率={selected['coverage']}")
            gated = block["test"]["gated"]
            print(f"          gated G={gated['mean']:+.4f}±{gated['se']:.4f} "
                  f"区间[{gated['ci_low']:+.4f}, {gated['ci_high']:+.4f}] "
                  f"({gated['improved_groups']}/{gated['n_test_files']} 文件为正) "
                  f"em1={gated['em1']} vs 基线 {block['test']['baseline']['em1']}")
        print(f"          判据结论:{verdict['conclusion']} —— {verdict['reason']}")

    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[输出] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
