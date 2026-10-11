"""增量 4 端到端:预算扫描(Q1)+ 双池实验(C1)+ 报告与曲线。

用法:
    python scripts/bench_inc4.py --config configs/inc4.yaml
    python scripts/bench_inc4.py --config configs/inc4.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import (  # noqa: E402
    plot_budget_scan,
    plot_pool_experiment,
    run_budget_scan,
    run_pool_experiment,
)
from everyngram.hf import HFModel, load_tokenizer, tokenizer_fingerprint  # noqa: E402
from everyngram.memory import Memory  # noqa: E402
from everyngram.select import (  # noqa: E402
    collect_contexts,
    effective_k,
    overlap_curve,
    score_contexts,
    select_pools,
)
from everyngram.tasks import (  # noqa: E402
    build_completion_task,
    build_corpus_ids,
    collect_code_files,
    split_files,
)

DEFAULT_CONFIG = Path("configs/inc4.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="压缩规模跑一遍链路:1 个预算、16 个上下文、4 个用例、1 次重复",
    )
    return parser.parse_args()


def score_summary(scored: list[dict]) -> dict:
    """两个分数在可用候选上的分布;缺失位置单独计数。"""
    import numpy as np

    usable = [
        item
        for item in scored
        if item["speed_score"] is not None and item["quality_score"] is not None
    ]
    missing = len(scored) - len(usable)
    if not usable:
        return {"contexts": len(scored), "candidates": 0, "missing": missing}
    speed = sorted(item["speed_score"] for item in usable)
    quality = sorted(item["quality_score"] for item in usable)
    middle = len(usable) // 2
    speed_ranks = np.argsort(np.argsort([item["speed_score"] for item in usable]))
    quality_ranks = np.argsort(np.argsort([item["quality_score"] for item in usable]))
    rank_corr = float(np.corrcoef(speed_ranks, quality_ranks)[0, 1]) if len(usable) > 2 else None
    same_draft = [item for item in usable if item["draft_first"] == item["true_next"]]
    diff_draft = [item for item in usable if item["draft_first"] != item["true_next"]]

    def subset_corr(subset: list[dict]):
        if len(subset) < 3:
            return None
        s = np.argsort(np.argsort([item["speed_score"] for item in subset]))
        q = np.argsort(np.argsort([item["quality_score"] for item in subset]))
        return round(float(np.corrcoef(s, q)[0, 1]), 4)

    return {
        "contexts": len(scored),
        "candidates": len(usable),
        "missing": missing,
        "speed_median": round(speed[middle], 6),
        "speed_max": round(speed[-1], 6),
        "quality_median": round(quality[middle], 6),
        "quality_max": round(quality[-1], 6),
        "speed_quality_rank_corr": None if rank_corr is None else round(rank_corr, 4),
        "draft_equals_true": len(same_draft),
        "draft_equals_true_rate": round(len(same_draft) / len(usable), 4),
        "rank_corr_draft_eq_true": subset_corr(same_draft),
        "rank_corr_draft_ne_true": subset_corr(diff_draft),
    }


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke = bool(args.smoke)

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    budgets = tuple(cfg["budgets"])
    max_files = int(cfg["max_files"])
    n_contexts = int(cfg["n_contexts"])
    n_cases = int(cfg["n_cases"])
    repeats = int(cfg["repeats"])
    if smoke:
        budgets = (int(budgets[0]),)
        max_files = min(max_files, 8)
        n_contexts = min(n_contexts, 16)
        n_cases = min(n_cases, 4)
        repeats = 1

    tokenizer = load_tokenizer(cfg["model"])
    fingerprint = tokenizer_fingerprint(tokenizer)

    files = collect_code_files([cfg["corpus"]], max_files=max_files)
    if len(files) < 4:
        raise ValueError(f"可用源文件太少({len(files)}),无法切分建表/开发/评测三份")
    corpus_files, holdout_files = split_files(files)
    dev_files, test_files = split_files(holdout_files)
    corpus_ids = build_corpus_ids(corpus_files, tokenizer)
    print(
        f"[切分] 文件 {len(files)}:建表 {len(corpus_files)} / 开发 {len(dev_files)} "
        f"/ 评测 {len(test_files)};语料 token {len(corpus_ids)}"
    )

    memory_full = Memory.build(
        corpus_ids, fingerprint=fingerprint, orders=(1, 2, 3), budget=1_000_000
    )
    model = HFModel(cfg["model"])
    task = build_completion_task(test_files, tokenizer, n_cases=n_cases)
    contexts = collect_contexts(dev_files, tokenizer, n=n_contexts, context_len=128)
    print(f"[用例] 评测 {len(task.cases)} 个;开发上下文 {len(contexts)} 个")

    scan = run_budget_scan(
        model,
        corpus_ids,
        fingerprint=fingerprint,
        budgets=budgets,
        cases=task.cases,
        lambdas=tuple(cfg["lambdas"]),
        gamma=int(cfg["gamma"]),
        max_new_tokens=int(cfg["max_new_tokens"]),
        repeats=repeats,
        out_dir=out.parent,
    )
    print(f"[Q1] 预算扫描完成:{len(scan['rows'])} 行")

    scored = score_contexts(model, memory_full, contexts)
    summary = score_summary(scored)
    k_used = effective_k(scored, int(cfg["k"]))
    pools = select_pools(scored, k=k_used)
    candidates = summary["candidates"]
    ks = sorted({max(1, round(candidates * frac)) for frac in (0.03, 0.06, 0.16, 0.32, 0.5)})
    curve = overlap_curve(scored, ks=ks)
    print(
        f"[C1] 候选 {candidates} 个;草稿=真值 {summary['draft_equals_true_rate']};"
        f"k {int(cfg['k'])} → {k_used};交叠 {curve['overlap']} at k={curve['ks']}"
    )

    pool_result = run_pool_experiment(
        model,
        memory_full,
        pools,
        cases=task.cases,
        gamma=int(cfg["gamma"]),
        lambdas=tuple(cfg["lambdas"]),
        max_new_tokens=int(cfg["max_new_tokens"]),
        repeats=repeats,
    )
    print(f"[C1] 池评测完成:{len(pool_result['pools'])} 个池")

    pool_block = {
        "scores": summary,
        "overlap": pools["overlap"],
        "overlap_count": pools["overlap_count"],
        "overlap_curve": curve,
        "k_effective": pools["k_effective"],
        "requested_k": int(cfg["k"]),
        **pool_result,
    }
    report = {
        "config": {
            "model": cfg["model"],
            "corpus": cfg["corpus"],
            "budgets": list(budgets),
            "lambdas": list(cfg["lambdas"]),
            "gamma": int(cfg["gamma"]),
            "k": int(cfg["k"]),
            "smoke": smoke,
        },
        "splits": {
            "files_total": len(files),
            "files_corpus": len(corpus_files),
            "files_dev": len(dev_files),
            "files_test": len(test_files),
            "corpus_tokens": len(corpus_ids),
            "full_entries": memory_full.stats()["entries"],
        },
        "budget_scan": scan,
        "pool_experiment": pool_block,
    }
    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plot_budget_scan(scan, out.parent / f"{out.name}_budget.png")
    plot_pool_experiment(pool_block, out.parent / f"{out.name}_pool.png")
    print(f"[输出] {json_path}、{out.name}_budget.png、{out.name}_pool.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
