"""增量 5 端到端:dev 选门槛与小 λ 网格,test 上评一次,出门控图。

用法:
    python scripts/bench_inc5.py --config configs/inc5.yaml
    python scripts/bench_inc5.py --config configs/inc5.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import plot_gated_experiment, run_gated_experiment  # noqa: E402
from everyngram.fuse import Gate  # noqa: E402
from everyngram.hf import HFModel, load_tokenizer, tokenizer_fingerprint  # noqa: E402
from everyngram.memory import Memory  # noqa: E402
from everyngram.tasks import (  # noqa: E402
    build_completion_task,
    build_corpus_ids,
    collect_code_files,
    split_files,
)

DEFAULT_CONFIG = Path("configs/inc5.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true", help="压规模跑链路")
    return parser.parse_args()


def build_gates(cfg: dict) -> tuple[Gate, ...]:
    return tuple(
        Gate(min_order=order, min_share=share, max_top_prob=top)
        for order in cfg["min_orders"]
        for share in cfg["min_shares"]
        for top in cfg["max_top_probs"]
    )


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke = bool(args.smoke)

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    max_files = int(cfg["max_files"])
    dev_cases = int(cfg["dev_cases"])
    test_cases = int(cfg["test_cases"])
    repeats = int(cfg["repeats"])
    gates = build_gates(cfg)
    lambdas = tuple(float(lam) for lam in cfg["lambdas"])
    if smoke:
        max_files = min(max_files, 8)
        dev_cases = min(dev_cases, 4)
        test_cases = min(test_cases, 4)
        repeats = 1
        gates = (Gate(min_order=1),)
        lambdas = lambdas[:1]

    tokenizer = load_tokenizer(cfg["model"])
    fingerprint = tokenizer_fingerprint(tokenizer)
    files = collect_code_files([cfg["corpus"]], max_files=max_files)
    if len(files) < 4:
        raise ValueError(f"可用源文件太少({len(files)}),无法切三份")
    corpus_files, holdout_files = split_files(files)
    dev_files, test_files = split_files(holdout_files)
    corpus_ids = build_corpus_ids(corpus_files, tokenizer)
    print(
        f"[切分] 文件 {len(files)}:建表 {len(corpus_files)} / 开发 {len(dev_files)} "
        f"/ 评测 {len(test_files)};语料 token {len(corpus_ids)}"
    )

    memory = Memory.build(
        corpus_ids, fingerprint=fingerprint, orders=tuple(cfg["orders"]), budget=1_000_000
    )
    model = HFModel(cfg["model"])
    dev_task = build_completion_task(dev_files, tokenizer, n_cases=dev_cases)
    test_task = build_completion_task(test_files, tokenizer, n_cases=test_cases)
    print(f"[用例] dev {len(dev_task.cases)} 个;test {len(test_task.cases)} 个")

    result = run_gated_experiment(
        model,
        memory,
        dev_cases=dev_task.cases,
        test_cases=test_task.cases,
        gates=gates,
        lambdas=lambdas,
        min_coverage=float(cfg["min_coverage"]),
        gamma=int(cfg["gamma"]),
        em_tokens=int(cfg["em_tokens"]),
        timing_tokens=int(cfg["timing_tokens"]),
        repeats=repeats,
    )
    best = result["selection"]["best"]
    print(
        "[dev] 选中 "
        + (f"gate={best['gate']} λ={best['lam']} 覆盖率={best['coverage']} "
           f"dev_gain={best['dev_gain']}" if best else "无(所有配置覆盖率不足)")
    )
    for name, block in result["test"]["configs"].items():
        delta = block["nll_delta"]
        print(
            f"[test] {name:>16}: G={delta['mean']:+.4f}±{delta['se']:.4f} "
            f"(改善 {delta['improved']}/{delta['n']}) acc={block['accuracy']} "
            f"EM={block['em']} 加速比={block['speedup']} 接受率={block['accept_rate']}"
        )

    report = {
        "config": {
            "model": cfg["model"],
            "corpus": cfg["corpus"],
            "gates": [gate.as_dict() for gate in gates],
            "lambdas": list(lambdas),
            "min_coverage": float(cfg["min_coverage"]),
            "smoke": smoke,
        },
        "splits": {
            "files_total": len(files),
            "files_corpus": len(corpus_files),
            "files_dev": len(dev_files),
            "files_test": len(test_files),
            "corpus_tokens": len(corpus_ids),
            "entries": memory.stats()["entries"],
            "dev_cases": len(dev_task.cases),
            "test_cases": len(test_task.cases),
        },
        **result,
    }
    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plot_gated_experiment(result, out.parent / f"{out.name}_gated.png")
    print(f"[输出] {json_path}、{out.name}_gated.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
