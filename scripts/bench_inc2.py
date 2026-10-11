"""增量 2 端到端:建表、跑读数、写报告与曲线。

用法:
    python scripts/bench_inc2.py --model data/models/Qwen3-0.6B --out reports/inc2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import (  # noqa: E402
    BenchReport,
    conclude,
    gpu_peak_mb,
    ledger,
    lookup_latency,
    measure_peak_rss,
    predict_speedup,
    reset_gpu_peak,
    run_decode_benchmark,
    table_readings,
)
from everyngram.hf import HFModel, load_tokenizer, tokenizer_fingerprint  # noqa: E402
from everyngram.memory import Memory  # noqa: E402
from everyngram.tasks import (  # noqa: E402
    build_completion_task,
    build_corpus_ids,
    collect_code_files,
    next_token_accuracy,
    perplexity,
    split_files,
)

PY_ROOTS = [r"data/python-stdlib"]
OOD_TEXT = Path("data/ood.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/Qwen3-0.6B")
    parser.add_argument("--out", default="reports/inc2")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gammas", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gammas = [int(x) for x in args.gammas.split(",") if x.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model)
    fingerprint = tokenizer_fingerprint(tokenizer)

    files = collect_code_files(PY_ROOTS, max_files=args.max_files)
    corpus_files, holdout_files = split_files(files)
    corpus_ids = build_corpus_ids(corpus_files, tokenizer)
    memory = Memory.build(
        corpus_ids, fingerprint=fingerprint, orders=(1, 2, 3), budget=1_000_000
    )
    memory_path = out.with_suffix(".ngmem")
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory.save(memory_path)

    model = HFModel(args.model)
    # 域内提示词:与增量 1 的结论一致,只有生成内容与语料重叠时草稿才可能被接受
    prompt_ids = corpus_ids[:32]

    reset_gpu_peak()
    degraded: list[str] = []
    try:
        r6, peak_rss_mb = measure_peak_rss(
            run_decode_benchmark,
            model,
            memory,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            gammas=gammas,
            repeats=args.repeats,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        reduced = [g for g in gammas if g <= max(1, max(gammas) // 2)] or [1]
        degraded.append(f"显存不足:gamma 从 {gammas} 降到 {reduced}")
        gammas = reduced
        reset_gpu_peak()
        r6, peak_rss_mb = measure_peak_rss(
            run_decode_benchmark,
            model,
            memory,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            gammas=gammas,
            repeats=args.repeats,
        )

    contexts = [
        case.prompt_ids
        for case in build_completion_task(holdout_files, tokenizer, n_cases=16).cases
    ]
    ledger_values = ledger(model, memory, contexts, gamma=max(gammas))
    predictions = [
        predict_speedup(ledger_values, r6["runs"][str(g)]["stats"]["accept_length"])
        for g in gammas
    ]

    task = build_completion_task(holdout_files, tokenizer, n_cases=16)
    ood_ids = tokenizer.encode(
        OOD_TEXT.read_text(encoding="utf-8", errors="ignore")[:4000],
        add_special_tokens=False,
    )

    readings = {
        "R1": table_readings(memory, memory_path),
        "R2": {str(g): r6["runs"][str(g)]["stats"] for g in gammas},
        "R3": lookup_latency(memory, contexts, repeats=10),
        "R4": {"peak_gpu_mb": gpu_peak_mb()},
        "R5": {"peak_rss_mb": peak_rss_mb},
        "R6": r6,
        "R7": {
            "in_domain": {
                "task": task.name,
                "cases": len(task.cases),
                "next_token_accuracy": round(next_token_accuracy(model, task.cases), 4),
            },
            "out_of_domain": {
                "perplexity": round(perplexity(model, ood_ids), 4),
            },
        },
    }
    account = {"ledger": ledger_values, "predictions": predictions}
    report = BenchReport(
        config=vars(args),
        env={
            "gammas": gammas,
            "fingerprint": fingerprint,
            "degraded": degraded,
            "prompt": "corpus_ids[:32]",
        },
        readings=readings,
        account=account,
        tasks={"code": {"cases": len(task.cases)}},
        conclusion=conclude(
            [run["speedup"] for run in r6["runs"].values()], consistency_ok=True
        ),
    )
    report.save_json(out.with_suffix(".json"))
    report.plot(out.with_suffix(".png"))
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2)[:4000])
    print(f"报告已写入 {out.with_suffix('.json')} 与 {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
