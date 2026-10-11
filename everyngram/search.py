"""eng search:在候选语料 × 预算之间按预注册规则挑一张表,并产出可复核的搜索报告。

规则(v1,冻结于 2026-10-10 的增量 22 设计):
- objective=task:在满足 OOD 约束的候选里取 dev mean prefix 最大;并列取更小
  λ → 更小预算 → 名称字典序;
- objective=speed:在满足接受长度下限的候选里取平均接受长度最大;并列取更小预算
  → 名称字典序;
- objective=both:两条可行性都要求满足,先按 task 再按 speed 排序。
可行集合为空是合法结果(selected 为 None)。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from everyngram.bench import (
    _prefix_score,
    run_alignment_probe,
    run_task_layer_experiment,
)
from everyngram.decode import greedy_generate
from everyngram.fuse import FusionSpec
from everyngram.hf import corpus_token_ids, tokenizer_fingerprint
from everyngram.memory import Memory
from everyngram.tasks import (
    build_corpus_ids,
    collect_code_files,
    collect_table_prompts,
    perplexity,
)

_OBJECTIVES = ("task", "speed", "both")


def _task_feasibility(row: dict, ood_max_ratio: float | None) -> tuple[bool, str]:
    if row.get("screened_out"):
        return False, f"被筛选: {row['screened_out']}"
    task = row.get("task")
    if task is None:
        return False, "缺任务读数"
    ratio = task.get("ood_ratio")
    if ood_max_ratio is not None:
        if ratio is None:
            return False, "缺 OOD 读数,无法判定 OOD 约束"
        if float(ratio) > float(ood_max_ratio):
            return False, f"OOD ×{float(ratio):.3f} 超过约束 ×{float(ood_max_ratio):.3f}"
    return True, "ok"


def _speed_feasibility(
    row: dict, accept_length_min: float | None
) -> tuple[bool, str]:
    if row.get("screened_out"):
        return False, f"被筛选: {row['screened_out']}"
    speed = row.get("speed")
    if speed is None or speed.get("accept_length") is None:
        return False, "缺速度读数"
    length = float(speed["accept_length"])
    if accept_length_min is not None and length < float(accept_length_min):
        return False, f"接受长度 {length:.3f} 低于下限 {float(accept_length_min):.3f}"
    return True, "ok"


def _task_key(row: dict) -> tuple:
    task = row["task"]
    return (
        -float(task["dev_prefix"]),
        float(task["lam"]),
        int(row["budget"]),
        str(row["name"]),
    )


def _speed_key(row: dict) -> tuple:
    return (
        -float(row["speed"]["accept_length"]),
        int(row["budget"]),
        str(row["name"]),
    )


def _both_key(row: dict) -> tuple:
    task = row["task"]
    speed = row["speed"]
    return (
        -float(task["dev_prefix"]),
        float(task["lam"]),
        -float(speed["accept_length"]),
        int(row["budget"]),
        str(row["name"]),
    )


def select_best(
    rows: list[dict],
    objective: str,
    *,
    ood_max_ratio: float | None = None,
    accept_length_min: float | None = None,
) -> dict:
    """按预注册规则选表;返回 {name, budget, reason, rejected, row}。"""
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective 只能是 {_OBJECTIVES}: {objective}")

    rejected: list[dict] = []
    feasible: list[dict] = []
    for row in rows:
        task_ok, task_reason = _task_feasibility(row, ood_max_ratio)
        speed_ok, speed_reason = _speed_feasibility(row, accept_length_min)
        if objective == "task":
            ok, reason = task_ok, task_reason
        elif objective == "speed":
            ok, reason = speed_ok, speed_reason
        else:
            ok = task_ok and speed_ok
            reasons = [
                value
                for value, passed in (
                    (task_reason, task_ok),
                    (speed_reason, speed_ok),
                )
                if not passed
            ]
            reason = "; ".join(reasons) if reasons else "ok"
        if ok:
            feasible.append(row)
        else:
            rejected.append(
                {"name": str(row["name"]), "budget": int(row["budget"]), "reason": reason}
            )

    if not feasible:
        return {
            "name": None,
            "budget": None,
            "reason": "可行集合为空:所有候选都被约束或读数缺失排除",
            "rejected": rejected,
            "row": None,
        }

    key = {"task": _task_key, "speed": _speed_key, "both": _both_key}[objective]
    best = min(feasible, key=key)
    return {
        "name": str(best["name"]),
        "budget": int(best["budget"]),
        "reason": f"objective={objective} 的规则胜出",
        "rejected": rejected,
        "row": best,
    }


def _load_target_scenarios(tokenizer, cfg):
    files = sorted(Path(cfg.target_corpus).rglob("*.txt"))
    if not files:
        raise ValueError(f"目标语料里没有 .txt: {cfg.target_corpus}")
    total = cfg.scenario_offset + cfg.dev_scenarios + cfg.test_scenarios
    prompts = collect_table_prompts(
        files,
        tokenizer,
        n=total,
        context_len=cfg.context_len,
        tail_chars=cfg.tail_chars,
        horizon=cfg.horizon,
    )
    if len(prompts) < total:
        raise ValueError(f"目标场景不足:需要 {total} 个(实得 {len(prompts)})")

    window = prompts[cfg.scenario_offset : total]

    def to_scenario(index: int, item: dict) -> dict:
        return {
            "name": f"target-{cfg.scenario_offset + index:02d}",
            "prompt_ids": item["prompt_ids"],
            "reference_ids": item["reference_ids"],
        }

    dev = [to_scenario(i, item) for i, item in enumerate(window[: cfg.dev_scenarios])]
    test = [
        to_scenario(cfg.dev_scenarios + i, item)
        for i, item in enumerate(window[cfg.dev_scenarios :])
    ]
    return dev, test


def _prompt_hit_rate(memory: Memory, scenarios) -> float:
    hits = sum(
        1 for scenario in scenarios if memory.lookup(list(scenario["prompt_ids"])) is not None
    )
    return hits / max(len(scenarios), 1)


def _task_dev_eval(model, memory, scenarios, *, lambdas, horizon, ood_ids, ood_max_ratio):
    """dev 段任务读数:逐 λ 的 mean prefix 与 OOD 比值;best 受 OOD 约束。"""
    plain_scores = []
    for scenario in scenarios:
        output, _ = greedy_generate(
            model, scenario["prompt_ids"], max_new_tokens=horizon, use_cache=True
        )
        plain_scores.append(_prefix_score(output, scenario["reference_ids"]))
    plain_prefix = float(np.mean(plain_scores))

    base_ppl = perplexity(model, list(ood_ids)) if len(ood_ids) >= 2 else None
    per_lambda = []
    for lam in lambdas:
        scores = []
        for scenario in scenarios:
            output, _ = greedy_generate(
                model,
                scenario["prompt_ids"],
                max_new_tokens=horizon,
                fusion=FusionSpec(lam=lam, memory=memory),
                use_cache=True,
            )
            scores.append(_prefix_score(output, scenario["reference_ids"]))
        ratio = None
        if base_ppl:
            fused_ppl = perplexity(
                model, list(ood_ids), fusion=FusionSpec(lam=lam, memory=memory)
            )
            ratio = float(fused_ppl / base_ppl)
        per_lambda.append(
            {
                "lam": float(lam),
                "mean_prefix": float(np.mean(scores)),
                "ood_ratio": None if ratio is None else round(ratio, 4),
            }
        )

    feasible = [
        item
        for item in per_lambda
        if ood_max_ratio is None
        or (item["ood_ratio"] is not None and item["ood_ratio"] <= float(ood_max_ratio))
    ]
    best = (
        max(feasible, key=lambda item: (item["mean_prefix"], -item["lam"]))
        if feasible
        else None
    )
    return {"plain_prefix": plain_prefix, "per_lambda": per_lambda, "best": best}


def _speed_stats(model, memory, scenarios, *, gamma, max_new_tokens, check_lossless):
    lengths, accept_rates, hit_rates = [], [], []
    identical = True
    for scenario in scenarios:
        if check_lossless:
            plain, _ = greedy_generate(
                model,
                scenario["prompt_ids"],
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        output, stats = greedy_generate(
            model,
            scenario["prompt_ids"],
            max_new_tokens=max_new_tokens,
            memory=memory,
            gamma=gamma,
            use_cache=True,
        )
        if check_lossless and output != plain:
            identical = False
        lengths.append(float(stats.accept_length))
        accept_rates.append(float(stats.accept_rate))
        hit_rates.append(float(stats.hit_rate))
    n = len(lengths)
    mean = float(np.mean(lengths)) if n else 0.0
    se = float(np.std(lengths, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "accept_length": mean,
        "se": se,
        "mean_accept_rate": float(np.mean(accept_rates)) if n else 0.0,
        "hit_rate": float(np.mean(hit_rates)) if n else 0.0,
        "n": n,
        "all_identical": identical,
    }


def run_search(model, tokenizer, cfg) -> dict:
    """执行四段搜索;写 out_dir/search.json 与 out_dir/selected.ngmem。"""
    started = time.perf_counter()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = tokenizer_fingerprint(tokenizer)
    dev, test = _load_target_scenarios(tokenizer, cfg)
    ood_root = Path(cfg.ood_root)
    if not ood_root.exists():
        raise FileNotFoundError(
            f"OOD 语料根目录不存在: {ood_root.resolve()}(配置值 {cfg.ood_root})"
        )
    ood_files = collect_code_files(
        [str(ood_root)],
        max_files=cfg.ood_files,
        seed=cfg.seed,
        exclude_substrings=cfg.ood_exclude,
    )
    ood_ids = build_corpus_ids(ood_files, tokenizer)[: cfg.ood_max_tokens]
    print(
        f"[目标] {cfg.target_corpus}:dev {len(dev)} / test {len(test)};"
        f"OOD {len(ood_files)} 文件/{len(ood_ids)} token",
        flush=True,
    )

    rows: list[dict] = []
    cache: dict[tuple, Memory] = {}
    for candidate in cfg.candidates:
        corpus = Path(candidate["corpus"])
        if not corpus.exists():
            raise FileNotFoundError(f"候选语料不存在: {corpus.resolve()}")
        corpus_ids = corpus_token_ids(tokenizer, corpus)
        for budget in candidate["budgets"]:
            memory = Memory.build(
                corpus_ids,
                fingerprint=fingerprint,
                orders=cfg.orders,
                budget=int(budget),
                continuation_len=cfg.continuation_len,
            )
            stats = memory.stats()
            probe = run_alignment_probe(model, memory, dev, horizon=cfg.horizon)[
                "summary"
            ]
            row = {
                "name": str(candidate["name"]),
                "corpus": str(corpus),
                "budget": int(budget),
                "entries": int(stats["entries"]),
                "dropped": int(stats["dropped"]),
                "prompt_hit_rate": round(_prompt_hit_rate(memory, dev), 4),
                "a_model": float(probe["mean_a_model"]),
                "a_target": float(probe["mean_a_target"]),
                "probe_hit_rate": float(probe["mean_hit_rate"]),
            }
            screen_reasons = []
            if cfg.min_a_model is not None and row["a_model"] < cfg.min_a_model:
                screen_reasons.append(
                    f"a_model {row['a_model']:.3f} 低于筛选线 {cfg.min_a_model}"
                )
            if cfg.min_a_target is not None and row["a_target"] < cfg.min_a_target:
                screen_reasons.append(
                    f"a_target {row['a_target']:.3f} 低于筛选线 {cfg.min_a_target}"
                )
            if screen_reasons:
                row["screened_out"] = "; ".join(screen_reasons)
                row["task"] = None
                row["speed"] = None
                row["task_detail"] = None
            else:
                task = _task_dev_eval(
                    model,
                    memory,
                    dev,
                    lambdas=cfg.lambdas,
                    horizon=cfg.horizon,
                    ood_ids=ood_ids,
                    ood_max_ratio=cfg.ood_max_ratio,
                )
                row["task_detail"] = task
                row["task"] = (
                    None
                    if task["best"] is None
                    else {
                        "lam": task["best"]["lam"],
                        "dev_prefix": task["best"]["mean_prefix"],
                        "plain_prefix": task["plain_prefix"],
                        "ood_ratio": task["best"]["ood_ratio"],
                    }
                )
                row["speed"] = _speed_stats(
                    model,
                    memory,
                    dev,
                    gamma=cfg.gamma,
                    max_new_tokens=cfg.max_new_tokens,
                    check_lossless=False,
                )
            rows.append(row)
            cache[(row["name"], row["budget"])] = memory
            print(
                f"[候选] {row['name']}@{row['budget']} 条目 {row['entries']}"
                f" a_model={row['a_model']:.3f} a_target={row['a_target']:.3f}"
                f" task={'—' if row['task'] is None else row['task']['dev_prefix']}"
                f" speed={'—' if row['speed'] is None else round(row['speed']['accept_length'], 3)}",
                flush=True,
            )

    selection = select_best(
        rows,
        cfg.objective,
        ood_max_ratio=cfg.ood_max_ratio,
        accept_length_min=cfg.accept_length_min,
    )
    rejected = {(item["name"], item["budget"]): item["reason"] for item in selection["rejected"]}
    for row in rows:
        row["feasible_reason"] = rejected.get((row["name"], row["budget"]), "ok")

    report = {
        "config": {
            "model": cfg.model,
            "dtype": cfg.dtype,
            "objective": cfg.objective,
            "ood_max_ratio": cfg.ood_max_ratio,
            "accept_length_min": cfg.accept_length_min,
            "lambdas": list(cfg.lambdas),
            "gamma": cfg.gamma,
            "seed": cfg.seed,
            "max_new_tokens": cfg.max_new_tokens,
            "horizon": cfg.horizon,
        },
        "target": {
            "corpus": str(cfg.target_corpus),
            "scenario_offset": cfg.scenario_offset,
            "dev_scenarios": len(dev),
            "test_scenarios": len(test),
            "ood_files": len(ood_files),
            "ood_tokens": len(ood_ids),
        },
        "rows": rows,
        "selected": None,
        "test": None,
    }

    if selection["name"] is not None:
        winner = selection["row"]
        memory = cache[(selection["name"], selection["budget"])]
        memory.save(out_dir / "selected.ngmem")
        report["selected"] = {
            "name": selection["name"],
            "budget": selection["budget"],
            "reason": selection["reason"],
        }
        test_block: dict = {"name": selection["name"], "budget": selection["budget"]}
        if winner["task"] is not None:
            lam = float(winner["task"]["lam"])
            task_result = run_task_layer_experiment(
                model,
                memory,
                dev,
                test,
                lambdas=(lam,),
                horizon=cfg.horizon,
                ood_ids=ood_ids,
                bootstrap=cfg.bootstrap,
                seed=cfg.seed,
            )
            test_block["task"] = {"lam": lam, "paired": task_result["test_paired"]}
        else:
            test_block["task"] = None
        test_block["speed"] = _speed_stats(
            model,
            memory,
            test,
            gamma=cfg.gamma,
            max_new_tokens=cfg.max_new_tokens,
            check_lossless=True,
        )
        report["test"] = test_block
    else:
        report["selected"] = None
        report["reason"] = selection["reason"]

    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    path = out_dir / "search.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[输出] {path}", flush=True)
    return report
