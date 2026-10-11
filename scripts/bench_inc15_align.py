"""增量 15:三个载体上的对齐度探针(记忆↔模型 / 记忆↔目标)。

用法:
    python -u scripts/bench_inc15_align.py --config configs/inc15_align.yaml
    python -u scripts/bench_inc15_align.py --config configs/inc15_align.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import run_alignment_probe  # noqa: E402
from everyngram.hf import (  # noqa: E402
    HFModel,
    corpus_token_ids,
    load_tokenizer,
    tokenizer_fingerprint,
)
from everyngram.memory import Memory  # noqa: E402
from everyngram.tasks import collect_table_prompts  # noqa: E402

DEFAULT_CONFIG = Path("configs/inc15_align.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true", help="压规模跑链路")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke = bool(args.smoke)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = str(cfg.get("name_prefix", "inc15_align"))
    horizon = int(cfg["horizon"])
    if smoke:
        horizon = 8

    tokenizer = load_tokenizer(cfg["model"])
    fingerprint = tokenizer_fingerprint(tokenizer)
    model = HFModel(cfg["model"], dtype=str(cfg.get("dtype", "float32")))

    for carrier in cfg["carriers"]:
        name = str(carrier["name"])
        corpus = Path(str(carrier["corpus"]))
        if not corpus.exists():
            raise FileNotFoundError(f"载体目录不存在: {corpus.resolve()}")
        sample = 8 if smoke else int(carrier["table_sample"])
        memory = Memory.build(
            corpus_token_ids(tokenizer, corpus),
            fingerprint=fingerprint,
            orders=(1, 2, 3),
            budget=int(carrier["budget"]),
            continuation_len=16,
        )
        prompts = collect_table_prompts(
            sorted(corpus.rglob("*.txt")),
            tokenizer,
            n=sample,
            context_len=int(carrier["context_len"]),
            tail_chars=int(carrier["tail_chars"]),
            horizon=horizon,
        )
        scenarios = [
            {
                "name": f"{name}-{index:02d}",
                "prompt_ids": item["prompt_ids"],
                "reference_ids": item["reference_ids"],
            }
            for index, item in enumerate(prompts)
        ]
        if not scenarios:
            raise ValueError(f"{name}: 没有抽到表格场景")
        per_step = bool(cfg.get("per_step", False))
        result = run_alignment_probe(
            model, memory, scenarios, horizon=horizon, per_step=per_step
        )
        payload = {
            "config": {
                "carrier": name,
                "corpus": str(corpus),
                "budget": int(carrier["budget"]),
                "table_sample": sample,
                "horizon": horizon,
                "per_step": per_step,
                "smoke": smoke,
            },
            "memory": memory.stats(),
            **result,
        }
        path = out_dir / f"{name_prefix}_{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary = result["summary"]
        print(
            f"[{name}] entries={payload['memory']['entries']}"
            f" a_model={summary['mean_a_model']:.3f}(中位 {summary['median_a_model']:.3f})"
            f" a_target={summary['mean_a_target']:.3f}(中位 {summary['median_a_target']:.3f})"
            f" hit={summary['mean_hit_rate']:.3f} -> {path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
