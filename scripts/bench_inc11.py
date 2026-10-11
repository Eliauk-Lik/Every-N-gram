"""增量 11:在线探针的投机开关(留出场景)。

用法:
    python -u scripts/bench_inc11.py --config configs/inc11.yaml
    python -u scripts/bench_inc11.py --config configs/inc11.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import (  # noqa: E402
    plot_adaptive_speed,
    run_adaptive_speed_experiment,
)
from everyngram.hf import (  # noqa: E402
    HFModel,
    corpus_token_ids,
    load_tokenizer,
    tokenizer_fingerprint,
)
from everyngram.memory import Memory  # noqa: E402
from everyngram.tasks import collect_table_prompts  # noqa: E402

DEFAULT_CONFIG = Path("configs/inc11.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true", help="压规模跑链路")
    return parser.parse_args()


def verdict(summary: dict) -> dict:
    """按预注册判据 C11.1 到 C11.4 给出机器可读判定。"""
    out: dict = {}
    copy_ratio = summary.get("speedup_vs_plain_by_arm", {}).get("copy", {})
    copy_ci = summary.get("speedup_vs_plain_ci_by_arm", {}).get("copy", {})
    for policy in sorted(p for p in summary.get("totals", {}) if p.startswith("adaptive-")):
        ratio = copy_ratio.get(policy)
        ci = copy_ci.get(policy, {})
        paired = summary.get("vs_spec_paired", {}).get(policy, {})
        auc = summary.get("auc", {}).get(policy)
        out[policy] = {
            "C11.1_copy_speedup": ratio,
            "C11.1_ci": [ci.get("ci_low"), ci.get("ci_high")],
            "C11.1_passes": bool(
                ratio is not None
                and ratio > 1.0
                and ci.get("ci_low") is not None
                and ci["ci_low"] > 1.0
            ),
            "C11.2_vs_spec_ci_low": paired.get("ci_low"),
            "C11.2_passes": bool(
                paired.get("ci_low") is not None and paired["ci_low"] > 0.0
            ),
            "C11.3_all_identical": bool(summary.get("all_identical")),
            "C11.4_auc": auc,
            "C11.4_passes": bool(auc is not None and auc >= 0.65),
            "switched_fraction": summary.get("switched_fraction", {}).get(policy),
        }
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
    table_sample = int(cfg["table_sample"])
    held_out_from = int(cfg["held_out_from"])
    copy_scenarios = int(cfg["copy_scenarios"])
    probe_steps = tuple(int(k) for k in cfg["probe_steps"])
    max_new_tokens = int(cfg["max_new_tokens"])
    repeats = int(cfg["repeats"])
    free_prompts = [str(p) for p in cfg["free_prompts"]]
    if smoke:
        held_out_from = 16
        copy_scenarios = 2
        free_prompts = free_prompts[:2]
        probe_steps = (8,)
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

    all_prompts = collect_table_prompts(
        sorted(corpus_dir.rglob("*.txt")),
        tokenizer,
        n=table_sample,
        context_len=int(cfg["context_len"]),
        tail_chars=int(cfg["tail_chars"]),
    )
    held_out = all_prompts[held_out_from : held_out_from + copy_scenarios]
    scenarios = []
    for index, item in enumerate(held_out):
        scenarios.append(
            {
                "name": f"copy-holdout-{index}",
                "arm": "copy",
                "prompt_ids": item["prompt_ids"],
                "origin": item["origin"],
            }
        )
    for index, prompt in enumerate(free_prompts):
        scenarios.append(
            {
                "name": f"free-{index}",
                "arm": "free",
                "prompt_ids": tuple(
                    tokenizer.encode(prompt, add_special_tokens=False)
                ),
                "origin": "instruction",
            }
        )
    print(
        f"[场景] 留出表格 {len(held_out)} 个(样本 {table_sample} 个的第 "
        f"{held_out_from} 起),自由 {len(free_prompts)} 个;"
        f"probe_steps={probe_steps} gamma={cfg['gamma']} repeats={repeats}",
        flush=True,
    )
    if not held_out:
        raise ValueError("没有抽到留出表格场景:检查语料里的表格行数与抽样参数")

    result = run_adaptive_speed_experiment(
        model,
        memory,
        scenarios,
        gamma=int(cfg["gamma"]),
        max_new_tokens=max_new_tokens,
        repeats=repeats,
        probe_steps=probe_steps,
        min_agree=float(cfg["min_agree"]),
        bootstrap=int(cfg["bootstrap"]),
    )
    print(f"[总时间] {result['summary']['totals']}", flush=True)
    print(f"[相对 plain] {result['summary']['speedup_vs_plain']}", flush=True)
    print(
        f"[按臂] {result['summary']['speedup_vs_plain_by_arm']}",
        flush=True,
    )
    result["verdict"] = verdict(result["summary"])
    print(f"[判据] {result['verdict']}", flush=True)

    report = {
        "config": {
            "model": cfg["model"],
            "dtype": cfg.get("dtype", "float32"),
            "corpus": str(corpus_dir),
            "table_sample": table_sample,
            "held_out_from": held_out_from,
            "copy_scenarios": len(held_out),
            "free_prompts": len(free_prompts),
            "gamma": int(cfg["gamma"]),
            "max_new_tokens": max_new_tokens,
            "repeats": repeats,
            "probe_steps": list(probe_steps),
            "min_agree": float(cfg["min_agree"]),
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
    plot_adaptive_speed(result, png_path)
    print(f"[输出] {json_path} 与 {png_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
