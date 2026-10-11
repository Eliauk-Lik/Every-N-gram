"""增量 15:对齐度-收益曲线汇总(三载体 × 两出口)。

用法:
    python -u scripts/bench_inc15_curve.py --config configs/inc15_curve.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.bench import _spearman  # noqa: E402

DEFAULT_CONFIG = Path("configs/inc15_curve.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = []
    for carrier in cfg["carriers"]:
        align = _load(str(carrier["align"]))
        task = _load(str(carrier["task"]))
        speed = _load(str(carrier["speed"]))
        rows.append(
            {
                "name": str(carrier["name"]),
                "a_model": float(align["summary"]["mean_a_model"]),
                "a_target": float(align["summary"]["mean_a_target"]),
                "task_benefit": float(task["test_paired"]["mean"]),
                "task_t": task["test_paired"]["t"],
                "speed_benefit": float(
                    speed["summary"]["speedup_vs_plain_by_arm"]["copy"]["spec"]
                ),
                "memory_entries": int(align["memory"]["entries"]),
            }
        )
        print(
            f"[{carrier['name']}] a_model={rows[-1]['a_model']:.3f}"
            f" a_target={rows[-1]['a_target']:.3f}"
            f" task={rows[-1]['task_benefit']:+.4f}(t={rows[-1]['task_t']})"
            f" speed(copy,gamma8)={rows[-1]['speed_benefit']:.3f}",
            flush=True,
        )

    a_models = [row["a_model"] for row in rows]
    a_targets = [row["a_target"] for row in rows]
    task_benefits = [row["task_benefit"] for row in rows]
    speed_benefits = [row["speed_benefit"] for row in rows]
    rho_task = _spearman(a_targets, task_benefits)
    rho_speed = _spearman(a_models, speed_benefits)
    verdict = {
        "V1_non_degenerate": {
            "a_model_range": round(max(a_models) - min(a_models), 4),
            "a_target_range": round(max(a_targets) - min(a_targets), 4),
            "passes": bool(
                max(a_models) - min(a_models) >= 0.10
                and max(a_targets) - min(a_targets) >= 0.10
            ),
        },
        "V2_ordering": {
            "rho_task": None if rho_task is None else round(rho_task, 4),
            "rho_speed": None if rho_speed is None else round(rho_speed, 4),
            "passes": bool(rho_task == 1.0 and rho_speed == 1.0),
        },
        "V3_cross_structure": {
            "carriers": {
                row["name"]: {
                    "task_near_zero": abs(row["task_benefit"]) < 0.01,
                    "speed_above_one": row["speed_benefit"] > 1.0,
                    "a_target_low": row["a_target"] < min(a_targets) + 0.15,
                    "a_model_high": row["a_model"] > max(a_models) - 0.15,
                }
                for row in rows
            },
            "passes": any(
                row["a_target"] < 0.6
                and row["a_model"] > 0.6
                and abs(row["task_benefit"]) < 0.01
                and row["speed_benefit"] > 1.0
                for row in rows
            ),
        },
        "V4_note": "三点排序证据,不做回归拟合",
    }
    print(f"[判据] {verdict}", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    for row in rows:
        ax.scatter(row["a_target"], row["task_benefit"], s=60, color="#1f77b4")
        ax.annotate(
            row["name"],
            (row["a_target"], row["task_benefit"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("memory-target alignment (mean draft match)")
    ax.set_ylabel("task benefit (held-out prefix gain)")
    ax.set_title("task exit vs memory-target alignment")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for row in rows:
        ax.scatter(row["a_model"], row["speed_benefit"], s=60, color="#d62728")
        ax.annotate(
            row["name"],
            (row["a_model"], row["speed_benefit"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("memory-model alignment (mean step agreement)")
    ax.set_ylabel("speed benefit (copy arm, gamma=8)")
    ax.set_title("speed exit vs memory-model alignment")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = Path(str(cfg["out"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    payload = {"carriers": rows, "verdict": verdict}
    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[输出] {json_path} 与 {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
