"""S2: 读数采集、账本分解与报告。零机器学习依赖,重库全部延迟导入。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def table_readings(memory, path: str | Path | None = None) -> dict:
    """R1:表大小(条目数与文件字节数)。"""
    stats = memory.stats()
    size = Path(path).stat().st_size if path else None
    return {
        "entries": stats["entries"],
        "dropped": stats["dropped"],
        "by_order": stats["by_order"],
        "file_bytes": size,
        "file_mb": round(size / 1e6, 3) if size is not None else None,
    }


def lookup_latency(
    memory, contexts: Sequence[Sequence[int]], repeats: int = 5, batch: int = 50
) -> dict:
    """R3:查表延迟的 P50 与 P99,单位微秒。

    单次查表只有几十纳秒,低于 Windows 时钟分辨率(约 100ns),
    因此每个样本是 batch 次查表的平均值,避免大量 0 值样本。
    """
    samples: list[float] = []
    for _ in range(int(repeats)):
        for ctx in contexts:
            start = time.perf_counter_ns()
            for _ in range(int(batch)):
                memory.lookup(ctx)
            elapsed_ns = time.perf_counter_ns() - start
            samples.append(elapsed_ns / int(batch) / 1000.0)

    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(arr.size),
        "p50_us": round(float(np.percentile(arr, 50)), 3),
        "p99_us": round(float(np.percentile(arr, 99)), 3),
    }


def measure_latency(fn, repeats: int = 3, warmup: int = 1) -> dict:
    """预热后重复计时,返回中位数与极值(秒)。"""
    for _ in range(int(warmup)):
        fn()
    times: list[float] = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "repeats": int(repeats),
        "median_s": float(np.median(arr)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def measure_peak_rss(fn, *args, sample_interval: float = 0.05, **kwargs):
    """跑 fn 并后台采样进程 RSS,返回 (fn 的返回值, 峰值 MB)。"""
    import threading

    import psutil

    process = psutil.Process()
    peak = {"value": process.memory_info().rss}
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            peak["value"] = max(peak["value"], process.memory_info().rss)
            stop.wait(sample_interval)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        stop.set()
        thread.join(timeout=1.0)
        peak["value"] = max(peak["value"], process.memory_info().rss)
    return result, round(peak["value"] / 1e6, 1)


def reset_gpu_peak() -> None:
    """重置 CUDA 峰值统计;无 CUDA 时静默跳过。"""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float | None:
    """R4:峰值显存 MB;无 CUDA 时返回 None。"""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e6, 1)


def run_decode_benchmark(
    model,
    memory,
    prompt_ids,
    *,
    max_new_tokens: int,
    gammas: Sequence[int],
    repeats: int = 3,
) -> dict:
    """R6:基线与各 gamma 的每 token 延迟(中位数),含加速比。"""
    from everyngram.decode import greedy_generate

    baseline = measure_latency(
        lambda: greedy_generate(
            model, prompt_ids, max_new_tokens=max_new_tokens, use_cache=True
        ),
        repeats=repeats,
    )
    baseline_ms = baseline["median_s"] * 1000.0 / max(max_new_tokens, 1)

    runs: dict[str, dict] = {}
    for gamma in gammas:
        timing = measure_latency(
            lambda g=gamma: greedy_generate(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                memory=memory,
                gamma=g,
                use_cache=True,
            ),
            repeats=repeats,
        )
        _, stats = greedy_generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            memory=memory,
            gamma=gamma,
            use_cache=True,
        )
        ms_per_token = timing["median_s"] * 1000.0 / max(max_new_tokens, 1)
        runs[str(gamma)] = {
            "gamma": int(gamma),
            "median_s": timing["median_s"],
            "ms_per_token": round(ms_per_token, 4),
            "speedup": round(baseline_ms / ms_per_token, 4) if ms_per_token else None,
            "stats": stats.as_dict(),
        }

    return {
        "baseline_ms_per_token": round(baseline_ms, 4),
        "baseline_timing": baseline,
        "runs": runs,
    }


def run_copy_speed_experiment(
    model,
    memory,
    scenarios,
    *,
    gammas=(2, 4, 8),
    max_new_tokens: int = 32,
    repeats: int = 3,
    bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """场景 × gamma 的缓存通路速度读数(场景级聚类)。

    scenarios 是 {"name","arm","prompt_ids","origin"} 序列;每个场景先测缓存
    贪婪基线,再对每个 gamma 测投机解码,同时记录接受长度、命中率与逐位无损。
    """
    from everyngram.decode import greedy_generate
    from everyngram.tasks import clustered_delta_stats

    gammas = tuple(int(g) for g in gammas)
    rows: list[dict] = []
    for index, scenario in enumerate(scenarios):
        name = str(scenario.get("name") or f"scenario-{index}")
        arm = str(scenario.get("arm") or "copy")
        prompt_ids = [int(t) for t in scenario["prompt_ids"]]
        if not prompt_ids:
            raise ValueError(f"场景 {name} 的 prompt_ids 为空")
        baseline_output, _ = greedy_generate(
            model, prompt_ids, max_new_tokens=int(max_new_tokens), use_cache=True
        )
        baseline_timing = measure_latency(
            lambda: greedy_generate(
                model,
                prompt_ids,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
            ),
            repeats=int(repeats),
        )
        baseline_ms = baseline_timing["median_s"] * 1000.0 / max(
            int(max_new_tokens), 1
        )
        for gamma in gammas:
            timing = measure_latency(
                lambda g=gamma: greedy_generate(
                    model,
                    prompt_ids,
                    max_new_tokens=int(max_new_tokens),
                    memory=memory,
                    gamma=g,
                    use_cache=True,
                ),
                repeats=int(repeats),
            )
            output, stats = greedy_generate(
                model,
                prompt_ids,
                max_new_tokens=int(max_new_tokens),
                memory=memory,
                gamma=gamma,
                use_cache=True,
            )
            ms_per_token = timing["median_s"] * 1000.0 / max(
                int(max_new_tokens), 1
            )
            rows.append(
                {
                    "name": name,
                    "arm": arm,
                    "origin": str(scenario.get("origin", "")),
                    "gamma": int(gamma),
                    "baseline_ms_per_token": round(baseline_ms, 4),
                    "ms_per_token": round(ms_per_token, 4),
                    "speedup": (
                        round(baseline_ms / ms_per_token, 4)
                        if ms_per_token > 0
                        else None
                    ),
                    "accept_length": round(stats.accept_length, 4),
                    "accept_rate": round(stats.accept_rate, 4),
                    "hit_rate": round(stats.hit_rate, 4),
                    "identical": output == baseline_output,
                    "steps": int(stats.steps),
                    "produced": int(stats.produced),
                }
            )

    summary: dict[str, dict] = {}
    for arm in sorted({row["arm"] for row in rows}):
        summary[arm] = {}
        for gamma in gammas:
            subset = [
                row
                for row in rows
                if row["arm"] == arm and row["gamma"] == int(gamma)
            ]
            if not subset:
                continue
            stats = clustered_delta_stats(
                [(row["name"], row["speedup"]) for row in subset],
                bootstrap=int(bootstrap),
                seed=seed,
            )
            speedups = [float(row["speedup"]) for row in subset]
            summary[arm][str(int(gamma))] = {
                "n_scenarios": len(subset),
                "mean": round(stats["mean"], 6),
                "se": round(stats["se"], 6),
                "ci_low": None if stats["ci_low"] is None else round(stats["ci_low"], 6),
                "ci_high": (
                    None if stats["ci_high"] is None else round(stats["ci_high"], 6)
                ),
                "median": round(float(np.median(speedups)), 6),
                "frac_above_one": round(
                    sum(1 for value in speedups if value > 1.0) / len(speedups), 4
                ),
                "mean_accept_length": round(
                    float(np.mean([row["accept_length"] for row in subset])), 6
                ),
                "mean_accept_rate": round(
                    float(np.mean([row["accept_rate"] for row in subset])), 6
                ),
                "mean_hit_rate": round(
                    float(np.mean([row["hit_rate"] for row in subset])), 6
                ),
                "all_identical": all(row["identical"] for row in subset),
            }
    return {"rows": rows, "summary": summary}


def _auc(scores, labels) -> float | None:
    """AUC(正值分数高于负值分数的概率);某一类为空时返回 None。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def run_adaptive_speed_experiment(
    model,
    memory,
    scenarios,
    *,
    gamma: int = 8,
    max_new_tokens: int = 32,
    repeats: int = 3,
    probe_steps=(8, 16),
    min_agree: float = 0.5,
    bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """三路径(普通/always-spec/在线探针自适应)的场景级时序与门控统计。"""
    import random

    from everyngram.decode import adaptive_greedy_generate, greedy_generate
    from everyngram.tasks import clustered_delta_stats

    gamma = int(gamma)
    probes = tuple(int(k) for k in probe_steps)
    policies = ["plain", "spec"] + [f"adaptive-k{k}" for k in probes]
    rows: list[dict] = []
    per_scenario: dict[str, dict] = {}
    for index, scenario in enumerate(scenarios):
        name = str(scenario.get("name") or f"scenario-{index}")
        arm = str(scenario.get("arm") or "copy")
        prompt_ids = [int(t) for t in scenario["prompt_ids"]]
        if not prompt_ids:
            raise ValueError(f"场景 {name} 的 prompt_ids 为空")

        plain_out, _ = greedy_generate(
            model, prompt_ids, max_new_tokens=int(max_new_tokens), use_cache=True
        )
        plain_ms = (
            measure_latency(
                lambda: greedy_generate(
                    model,
                    prompt_ids,
                    max_new_tokens=int(max_new_tokens),
                    use_cache=True,
                ),
                repeats=int(repeats),
            )["median_s"]
            * 1000.0
        )
        entries: dict[str, dict] = {
            "plain": {"ms": plain_ms, "output": plain_out, "stats": None, "probe": None}
        }

        spec_out, spec_stats = greedy_generate(
            model,
            prompt_ids,
            max_new_tokens=int(max_new_tokens),
            memory=memory,
            gamma=gamma,
            use_cache=True,
        )
        spec_ms = (
            measure_latency(
                lambda: greedy_generate(
                    model,
                    prompt_ids,
                    max_new_tokens=int(max_new_tokens),
                    memory=memory,
                    gamma=gamma,
                    use_cache=True,
                ),
                repeats=int(repeats),
            )["median_s"]
            * 1000.0
        )
        entries["spec"] = {"ms": spec_ms, "output": spec_out, "stats": spec_stats, "probe": None}

        for k in probes:
            adaptive_out, adaptive_stats = adaptive_greedy_generate(
                model,
                prompt_ids,
                max_new_tokens=int(max_new_tokens),
                memory=memory,
                gamma=gamma,
                probe_steps=k,
                min_agree=float(min_agree),
            )
            adaptive_ms = (
                measure_latency(
                    lambda kk=k: adaptive_greedy_generate(
                        model,
                        prompt_ids,
                        max_new_tokens=int(max_new_tokens),
                        memory=memory,
                        gamma=gamma,
                        probe_steps=kk,
                        min_agree=float(min_agree),
                    ),
                    repeats=int(repeats),
                )["median_s"]
                * 1000.0
            )
            entries[f"adaptive-k{k}"] = {
                "ms": adaptive_ms,
                "output": adaptive_out,
                "stats": adaptive_stats,
                "probe": k,
            }

        per_scenario[name] = {"arm": arm, "spec_accept": spec_stats.accept_length}
        for policy in policies:
            entry = entries[policy]
            stats = entry["stats"]
            probe_steps_value = None
            probe_rate = None
            switched = None
            if entry["probe"] is not None:
                probe_steps_value = int(stats.probe_steps)
                probe_rate = (
                    stats.probe_agreed / stats.probe_steps
                    if stats.probe_steps
                    else 0.0
                )
                switched = bool(stats.switched)
            rows.append(
                {
                    "name": name,
                    "arm": arm,
                    "policy": policy,
                    "probe_steps": probe_steps_value,
                    "ms": round(float(entry["ms"]), 4),
                    "ms_per_token": round(
                        float(entry["ms"]) / max(int(max_new_tokens), 1), 4
                    ),
                    "identical": entry["output"] == plain_out,
                    "accept_length": (
                        round(float(stats.accept_length), 4) if stats else 1.0
                    ),
                    "accept_rate": (
                        round(float(stats.accept_rate), 4) if stats else None
                    ),
                    "hit_rate": round(float(stats.hit_rate), 4) if stats else None,
                    "probe_rate": (
                        round(float(probe_rate), 4) if probe_rate is not None else None
                    ),
                    "switched": switched,
                    "steps": int(stats.steps) if stats else None,
                    "produced": int(stats.produced) if stats else None,
                }
            )

    by_policy: dict[str, list[dict]] = {policy: [] for policy in policies}
    for row in rows:
        by_policy[row["policy"]].append(row)
    totals = {
        policy: round(sum(row["ms"] for row in by_policy[policy]), 4)
        for policy in policies
    }
    plain_total = totals["plain"]
    speedup_vs_plain = {
        policy: (round(plain_total / totals[policy], 4) if totals[policy] else None)
        for policy in policies
    }

    rng = random.Random(seed)
    scenario_names = sorted({row["name"] for row in rows})

    def bootstrap_ratio(policy: str, names: list[str]) -> dict:
        plain_ms = {
            row["name"]: row["ms"]
            for row in by_policy["plain"]
            if row["name"] in names
        }
        policy_ms = {
            row["name"]: row["ms"]
            for row in by_policy[policy]
            if row["name"] in names
        }
        if not names or plain_total <= 0:
            return {"ci_low": None, "ci_high": None, "bootstrap": int(bootstrap)}
        draws = []
        for _ in range(int(bootstrap)):
            sample = [names[rng.randrange(len(names))] for _ in names]
            base = sum(plain_ms[name] for name in sample)
            value = sum(policy_ms[name] for name in sample)
            draws.append(base / value if value > 0 else 0.0)
        draws.sort()
        return {
            "ci_low": round(float(draws[int(0.025 * len(draws))]), 4),
            "ci_high": round(float(draws[min(len(draws) - 1, int(0.975 * len(draws)))]), 4),
            "bootstrap": int(bootstrap),
        }

    speedup_ci = {
        policy: bootstrap_ratio(policy, list(scenario_names)) for policy in policies
    }
    arms = sorted({row["arm"] for row in rows})
    speedup_ci_by_arm = {
        arm: {
            policy: bootstrap_ratio(
                policy,
                [row["name"] for row in by_policy["plain"] if row["arm"] == arm],
            )
            for policy in policies
        }
        for arm in arms
    }
    speedup_by_arm: dict[str, dict] = {}
    for arm in arms:
        arm_plain = sum(
            row["ms"] for row in by_policy["plain"] if row["arm"] == arm
        )
        speedup_by_arm[arm] = {}
        for policy in policies:
            arm_total = sum(
                row["ms"] for row in by_policy[policy] if row["arm"] == arm
            )
            speedup_by_arm[arm][policy] = (
                round(arm_plain / arm_total, 4) if arm_total else None
            )

    vs_spec_paired: dict[str, dict] = {}
    for k in probes:
        policy = f"adaptive-k{k}"
        spec_ms = {
            row["name"]: row["ms"]
            for row in by_policy["spec"]
        }
        pairs = [
            (row["name"], spec_ms[row["name"]] - row["ms"])
            for row in by_policy[policy]
        ]
        stats = clustered_delta_stats(pairs, bootstrap=int(bootstrap), seed=seed)
        vs_spec_paired[policy] = {
            "mean": round(stats["mean"], 6),
            "se": round(stats["se"], 6),
            "ci_low": None if stats["ci_low"] is None else round(stats["ci_low"], 6),
            "ci_high": None if stats["ci_high"] is None else round(stats["ci_high"], 6),
        }

    auc = {}
    switched_fraction = {}
    for k in probes:
        policy = f"adaptive-k{k}"
        subset = by_policy[policy]
        auc[policy] = _auc(
            [row["probe_rate"] or 0.0 for row in subset],
            [per_scenario[row["name"]]["spec_accept"] > 2.3 for row in subset],
        )
        switched_fraction[policy] = round(
            sum(1 for row in subset if row["switched"]) / max(len(subset), 1), 4
        )

    summary = {
        "totals": totals,
        "speedup_vs_plain": speedup_vs_plain,
        "speedup_vs_plain_ci": speedup_ci,
        "speedup_vs_plain_ci_by_arm": speedup_ci_by_arm,
        "speedup_vs_plain_by_arm": speedup_by_arm,
        "vs_spec_paired": vs_spec_paired,
        "auc": auc,
        "switched_fraction": switched_fraction,
        "all_identical": all(row["identical"] for row in rows),
        "n_scenarios": len(scenario_names),
    }
    return {"rows": rows, "summary": summary}


def _prefix_score(output, reference) -> float:
    matched = 0
    for produced, expected in zip(output, reference):
        if produced != expected:
            break
        matched += 1
    return matched / max(len(reference), 1)


def _spearman(xs, ys) -> float | None:
    """秩相关(平均秩处理并列);样本不足或常数序列返回 None。"""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2 or ys.size != xs.size:
        return None

    def rank(values):
        order = values.argsort()
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        sorted_values = values[order]
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
                j += 1
            if j > i:
                ranks[order[i : j + 1]] = (i + j) / 2.0
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom else None


def run_task_layer_experiment(
    model,
    memory,
    dev_scenarios,
    test_scenarios,
    *,
    lambdas=(0.1, 0.2, 0.3, 0.4, 0.5),
    horizon: int = 16,
    ood_ids=(),
    bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """复制型载体上的任务层剂量-反应:dev 选 λ,留出评一次,附 OOD 曲线。"""
    from everyngram.decode import greedy_generate
    from everyngram.fuse import FusionSpec
    from everyngram.tasks import clustered_delta_stats, perplexity

    lambdas = tuple(float(lam) for lam in lambdas)
    configs = [("plain", None)] + [
        (f"fused-l{lam}", lam) for lam in lambdas
    ]

    def run_split(scenarios, split: str) -> list[dict]:
        plain_outputs: dict[str, list[int]] = {}
        for scenario in scenarios:
            name = str(scenario.get("name"))
            output, _ = greedy_generate(
                model,
                [int(t) for t in scenario["prompt_ids"]],
                max_new_tokens=int(horizon),
                use_cache=True,
            )
            plain_outputs[name] = output
        rows: list[dict] = []
        for config, lam in configs:
            per_scenario = []
            for scenario in scenarios:
                name = str(scenario.get("name"))
                reference = [int(t) for t in scenario["reference_ids"]]
                if lam is None:
                    output = plain_outputs[name]
                else:
                    output, _ = greedy_generate(
                        model,
                        [int(t) for t in scenario["prompt_ids"]],
                        max_new_tokens=int(horizon),
                        fusion=FusionSpec(
                            lam=lam, mode="arithmetic", memory=memory, gate=None
                        ),
                        use_cache=True,
                    )
                base = plain_outputs[name]
                changed = (
                    sum(
                        1
                        for produced, reference_token in zip(output, base)
                        if produced != reference_token
                    )
                    / max(len(base), 1)
                    if lam is not None
                    else 0.0
                )
                per_scenario.append(
                    {
                        "name": name,
                        "origin": scenario.get("origin"),
                        "prefix": round(_prefix_score(output, reference), 4),
                        "exact": output[: len(reference)] == reference,
                        "changed": round(changed, 4),
                    }
                )
            prefixes = np.array([item["prefix"] for item in per_scenario])
            rows.append(
                {
                    "split": split,
                    "config": config,
                    "lam": lam,
                    "mean_prefix": round(float(prefixes.mean()), 6),
                    "se_prefix": round(
                        float(prefixes.std(ddof=1) / np.sqrt(len(prefixes)))
                        if len(prefixes) > 1
                        else 0.0,
                        6,
                    ),
                    "exact": int(sum(1 for item in per_scenario if item["exact"])),
                    "mean_changed": round(
                        float(np.mean([item["changed"] for item in per_scenario])), 6
                    ),
                    "per_scenario": per_scenario,
                }
            )
        return rows

    dev_rows = run_split(dev_scenarios, "dev")
    test_rows = run_split(test_scenarios, "test")
    fused_dev = [row for row in dev_rows if row["lam"] is not None]
    if not fused_dev:
        raise ValueError("lambdas 不能为空")
    best = max(fused_dev, key=lambda row: (row["mean_prefix"], -float(row["lam"])))
    plain_test = next(row for row in test_rows if row["lam"] is None)
    selected_test = next(row for row in test_rows if row["config"] == best["config"])
    diffs = [
        (
            selected.get("origin") or selected["name"],
            float(selected["prefix"]) - float(plain["prefix"]),
        )
        for selected, plain in zip(
            selected_test["per_scenario"], plain_test["per_scenario"]
        )
    ]
    stats = clustered_delta_stats(diffs, bootstrap=int(bootstrap), seed=seed)
    test_paired = {
        "lam": best["lam"],
        "mean": round(stats["mean"], 6),
        "se": round(stats["se"], 6),
        "ci_low": None if stats["ci_low"] is None else round(stats["ci_low"], 6),
        "ci_high": None if stats["ci_high"] is None else round(stats["ci_high"], 6),
        "t": (
            round(stats["mean"] / stats["se"], 3) if stats["se"] > 0 else None
        ),
        "n": len(diffs),
        "n_groups": stats["n_groups"],
    }
    dev_spearman = _spearman(
        [row["lam"] for row in fused_dev],
        [row["mean_prefix"] for row in fused_dev],
    )

    ood: list[dict] = []
    if len(ood_ids) >= 2:
        base_ppl = perplexity(model, list(ood_ids))
        for config, lam in configs[1:]:
            value = perplexity(
                model,
                list(ood_ids),
                fusion=FusionSpec(
                    lam=lam, mode="arithmetic", memory=memory, gate=None
                ),
            )
            ood.append(
                {
                    "config": config,
                    "lam": lam,
                    "perplexity": round(float(value), 4),
                    "ratio_vs_plain": round(float(value / base_ppl), 4),
                }
            )

    return {
        "rows": dev_rows + test_rows,
        "selected": {
            "config": best["config"],
            "lam": best["lam"],
            "dev_prefix": best["mean_prefix"],
            "dev_plain_prefix": next(
                row["mean_prefix"] for row in dev_rows if row["lam"] is None
            ),
        },
        "test_paired": test_paired,
        "dev_spearman": (
            None if dev_spearman is None else round(dev_spearman, 4)
        ),
        "ood": ood,
    }


def run_alignment_probe(
    model, memory, scenarios, *, horizon: int = 16, per_step: bool = False
) -> dict:
    """两个对齐度:记忆↔模型(逐步一致率)与记忆↔目标(draft/参考最长公共前缀)。

    per_step=True 时每行额外记录 `step_agreements`(逐步 0/1),
    供接受长度的位置相关风险分解使用;默认关闭以保持报告体积。
    """
    from everyngram.fuse import softmax

    rows: list[dict] = []
    for index, scenario in enumerate(scenarios):
        name = str(scenario.get("name") or f"scenario-{index}")
        prompt = [int(t) for t in scenario["prompt_ids"]]
        reference = [int(t) for t in scenario["reference_ids"]]
        logits, cache = model.forward(prompt)
        ctx = list(prompt)
        agreements = 0
        hits = 0
        steps = 0
        step_flags: list[int] = []
        for step in range(int(horizon)):
            p = softmax(logits[-1])
            nxt = int(p.argmax())
            hit = memory.lookup(ctx)
            agreed = 0
            if hit is not None:
                hits += 1
                entry = hit[1]
                if entry.continuation and int(entry.continuation[0]) == nxt:
                    agreements += 1
                    agreed = 1
            step_flags.append(agreed)
            steps += 1
            ctx.append(nxt)
            if step < int(horizon) - 1:
                logits, cache = model.forward([nxt], cache)
        draft = list(memory.draft(prompt, int(horizon)))
        matched = 0
        for produced, expected in zip(draft, reference):
            if produced != expected:
                break
            matched += 1
        row = {
            "name": name,
            "a_model": round(agreements / max(steps, 1), 4),
            "a_target": round(matched / max(len(reference), 1), 4),
            "hit_rate": round(hits / max(steps, 1), 4),
            "steps": steps,
        }
        if per_step:
            row["step_agreements"] = step_flags
        rows.append(row)

    def _mean(field: str) -> float:
        return float(np.mean([row[field] for row in rows])) if rows else 0.0

    def _median(field: str) -> float:
        return float(np.median([row[field] for row in rows])) if rows else 0.0

    return {
        "rows": rows,
        "summary": {
            "n_scenarios": len(rows),
            "mean_a_model": round(_mean("a_model"), 4),
            "median_a_model": round(_median("a_model"), 4),
            "mean_a_target": round(_mean("a_target"), 4),
            "median_a_target": round(_median("a_target"), 4),
            "mean_hit_rate": round(_mean("hit_rate"), 4),
        },
    }


def plot_task_layer(report: dict, path: str | Path) -> None:
    """任务层剂量-反应(前缀匹配 vs λ)+ OOD 困惑度代价。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(report.get("rows", []))
    ood = list(report.get("ood", []))
    splits = sorted({str(row["split"]) for row in rows})
    colors = {"dev": "#1f77b4", "test": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    for split in splits:
        fused = sorted(
            [
                row
                for row in rows
                if str(row["split"]) == split and row["lam"] is not None
            ],
            key=lambda row: float(row["lam"]),
        )
        if fused:
            ax.errorbar(
                [float(row["lam"]) for row in fused],
                [float(row["mean_prefix"]) for row in fused],
                yerr=[float(row["se_prefix"]) for row in fused],
                marker="o",
                capsize=3,
                color=colors.get(split, "gray"),
                label=f"{split} (fused)",
            )
        plain = next(
            (
                row
                for row in rows
                if str(row["split"]) == split and row["lam"] is None
            ),
            None,
        )
        if plain is not None:
            ax.axhline(
                float(plain["mean_prefix"]),
                color=colors.get(split, "gray"),
                linestyle="--",
                linewidth=1,
                label=f"{split} plain",
            )
    ax.set_xlabel("fusion strength lambda")
    ax.set_ylabel("prefix match vs document continuation")
    ax.set_title("task-layer dose-response (copy carrier)")
    ax.grid(alpha=0.25)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)

    ax = axes[1]
    if ood:
        ordered = sorted(ood, key=lambda row: float(row["lam"]))
        ax.plot(
            [float(row["lam"]) for row in ordered],
            [float(row["perplexity"]) for row in ordered],
            marker="s",
            color="black",
        )
        base = float(ordered[0]["perplexity"]) / max(
            float(ordered[0].get("ratio_vs_plain") or 1.0), 1e-9
        )
        ax.axhline(base, color="gray", linestyle="--", linewidth=1, label="plain")
        for row in ordered:
            ax.annotate(
                f"x{float(row['ratio_vs_plain']):.2f}",
                (float(row["lam"]), float(row["perplexity"])),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.legend(fontsize=8)
    ax.set_xlabel("fusion strength lambda")
    ax.set_ylabel("OOD perplexity (code sample)")
    ax.set_title("out-of-domain cost")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_adaptive_speed(report: dict, path: str | Path) -> None:
    """自适应投机的门控散点(probe_rate vs 接受长度)与总时间柱。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(report.get("rows", []))
    summary = dict(report.get("summary", {}))
    adaptive_policies = sorted(
        {str(row["policy"]) for row in rows if str(row["policy"]).startswith("adaptive-")},
        key=lambda name: int(name.split("-k")[1]),
    )
    spec_accept = {
        str(row["name"]): float(row["accept_length"])
        for row in rows
        if str(row["policy"]) == "spec"
    }
    colors = ("#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    from matplotlib.lines import Line2D

    for color, policy in zip(colors, adaptive_policies):
        subset = [row for row in rows if str(row["policy"]) == policy]
        for row in subset:
            y = spec_accept.get(str(row["name"]), float(row["accept_length"]))
            ax.scatter(
                float(row["probe_rate"] or 0.0),
                y,
                color=color,
                marker="o" if row["switched"] else "x",
                s=48,
            )
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axhline(2.3, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("probe agreement rate (first k steps)")
    ax.set_ylabel("always-spec accept length (gamma=8)")
    ax.set_title("gate signal vs accept length")
    ax.grid(alpha=0.25)
    legend_handles = [
        Line2D([], [], color=color, marker="o", linestyle="none", markersize=6, label=policy)
        for color, policy in zip(colors, adaptive_policies)
    ]
    legend_handles.extend(
        [
            Line2D([], [], color="gray", linestyle=":", label="probe threshold 0.5"),
            Line2D([], [], color="black", linestyle="--", label="break-even accept 2.3"),
            Line2D([], [], color="gray", marker="o", linestyle="none", markersize=6, label="filled = switched"),
            Line2D([], [], color="gray", marker="x", linestyle="none", markersize=6, label="x = stayed plain"),
        ]
    )
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=7)

    ax = axes[1]
    policies = ["plain", "spec"] + adaptive_policies
    totals = dict(summary.get("totals", {}))
    base = float(totals.get("plain") or 1.0)
    xs = np.arange(len(policies))
    heights = [float(totals.get(policy, 0.0)) / base for policy in policies]
    bars = ax.bar(xs, heights)
    for x, height in zip(xs, heights):
        ax.annotate(
            f"{height:.3f}",
            (x, height),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(max(0.0, min(heights) - 0.1) if heights else 0.0, max(heights + [1.0]) + 0.05)
    ax.set_xticks(xs)
    ax.set_xticklabels(policies, fontsize=8, rotation=15)
    ax.set_ylabel("total time relative to plain")
    ax.set_title("workload total time (lower is better)")
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_copy_speed(report: dict, path: str | Path) -> None:
    """复制型载体速度读数:接受长度 vs 加速比散点 + 各臂均值与区间。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(report.get("rows", []))
    summary = dict(report.get("summary", {}))
    arms = sorted({str(row["arm"]) for row in rows})
    gammas = sorted({int(row["gamma"]) for row in rows})
    colors = {"copy": "#1f77b4", "free": "#d62728"}
    marker_list = ("o", "s", "^", "D", "v", "P")
    markers = {gamma: marker_list[i % len(marker_list)] for i, gamma in enumerate(gammas)}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    for arm in arms:
        for gamma in gammas:
            subset = [
                row
                for row in rows
                if str(row["arm"]) == arm and int(row["gamma"]) == gamma
            ]
            if not subset:
                continue
            ax.scatter(
                [float(row["accept_length"]) for row in subset],
                [float(row["speedup"]) for row in subset],
                color=colors.get(arm, "gray"),
                marker=markers[gamma],
                s=42,
                label=f"{arm} gamma={gamma}",
            )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(2.08, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("accept length (accepted tokens / step)")
    ax.set_ylabel("speedup (x, cached path)")
    ax.set_title("speedup vs accept length (per scenario)")
    ax.grid(alpha=0.25)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.8 / max(len(gammas), 1)
    x = np.arange(len(arms))
    for i, gamma in enumerate(gammas):
        means, lows, highs = [], [], []
        for arm in arms:
            block = summary.get(arm, {}).get(str(gamma))
            if not block:
                means.append(0.0)
                lows.append(0.0)
                highs.append(0.0)
                continue
            mean = float(block["mean"])
            means.append(mean)
            ci_low, ci_high = block.get("ci_low"), block.get("ci_high")
            lows.append(max(0.0, mean - float(ci_low)) if ci_low is not None else 0.0)
            highs.append(max(0.0, float(ci_high) - mean) if ci_high is not None else 0.0)
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=[lows, highs],
            capsize=3,
            label=f"gamma={gamma}",
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x + width * (len(gammas) - 1) / 2 if gammas else x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("mean speedup (95% CI)")
    ax.set_title("mean speedup by arm")
    ax.grid(alpha=0.25, axis="y")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class _NullModel:
    """零成本模型,只用来量解码循环自身的开销。"""

    def __init__(self, vocab: int = 8) -> None:
        self.logits = np.zeros((1, vocab), dtype=np.float64)

    def forward(self, token_ids, cache=None):
        return self.logits, None

    def crop(self, cache, keep: int):
        return cache


def loop_overhead_ms(max_new_tokens: int = 200) -> float:
    """o:解码循环每步的纯 Python 开销(不含模型前向)。"""
    from everyngram.decode import _plain_greedy_cached

    model = _NullModel()
    start = time.perf_counter()
    _plain_greedy_cached(model, [0], max_new_tokens)
    return (time.perf_counter() - start) * 1000.0 / max(max_new_tokens, 1)


def ledger(model, memory, contexts, *, gamma: int, repeats: int = 5) -> dict:
    """P4 账本:逐项测 c(1)、c(gamma)、o、d(gamma),单位毫秒。

    c(1) 与 c(gamma) 的主口径是**就地测量**:在真实上下文缓存上分别喂 1 个与
    gamma 个新 token(对应普通解码的一步与投机验证的一轮);无缓存口径保留为
    `*_nocache_*` 对照——增量 10 证明无缓存口径会低估每轮真实成本。
    """
    import time as _time
    from copy import deepcopy

    token = 7
    context = [int(t) for t in (contexts[0] if contexts else [])]
    if not context:
        context = [token] * 16

    c1_nocache = measure_latency(
        lambda: model.forward([token]), repeats=repeats
    )["median_s"]
    c_gamma_nocache = measure_latency(
        lambda: model.forward([token] * int(gamma)), repeats=repeats
    )["median_s"]

    _, base_cache = model.forward(list(context))

    def cached_latency(n_tokens: int) -> float:
        samples = []
        for _ in range(max(int(repeats), 1)):
            cache = deepcopy(base_cache)
            start = _time.perf_counter()
            model.forward([token] * int(n_tokens), cache)
            samples.append(_time.perf_counter() - start)
        return float(np.median(samples))

    cached_latency(1)  # 预热
    c1 = cached_latency(1)
    c_gamma = cached_latency(int(gamma))
    sweep = measure_latency(
        lambda: [memory.draft(ctx, gamma) for ctx in contexts], repeats=repeats
    )["median_s"] / max(len(contexts), 1)
    o = loop_overhead_ms() / 1000.0

    return {
        "c1_s": c1,
        "c_gamma_s": c_gamma,
        "c1_nocache_s": c1_nocache,
        "c_gamma_nocache_s": c_gamma_nocache,
        "o_s": o,
        "d_gamma_s": sweep,
        "context_len": len(context),
        "c1_ms": round(c1 * 1000.0, 4),
        "c_gamma_ms": round(c_gamma * 1000.0, 4),
        "c1_nocache_ms": round(c1_nocache * 1000.0, 4),
        "c_gamma_nocache_ms": round(c_gamma_nocache * 1000.0, 4),
        "o_ms": round(o * 1000.0, 4),
        "d_gamma_ms": round(sweep * 1000.0, 4),
    }


def predict_speedup(
    values: dict, accept_length: float, *, merged_bonus: bool = True
) -> dict:
    """按账本预测每 token 延迟与加速比。

    merged_bonus=True(默认,增量 23 起)对应"修正 token 并入下一轮验证批"的循环:
    每轮成本 = c(gamma+1) + o + d;False 保留旧循环的"验证 + bonus"两次前向口径。
    c(n) 在缓存通路上对 n 平坦,因此合并批与 c(gamma) 同量级。
    """
    plain = values["c1_s"] + values["o_s"]
    round_cost = values["c_gamma_s"] + values["o_s"] + values["d_gamma_s"]
    if not merged_bonus:
        round_cost += values["c1_s"]
    spec = round_cost / max(float(accept_length), 1e-9)
    return {
        "plain_ms_per_token": round(plain * 1000.0, 4),
        "spec_ms_per_token": round(spec * 1000.0, 4),
        "predicted_speedup": round(plain / spec, 4) if spec > 0 else None,
    }


def conclude(speedups: Sequence[float | None], *, consistency_ok: bool) -> str:
    """三态判定:不成立(一致性失败或缺读数)/ 通过(有加速)/ 负结果(读数齐但无加速)。"""
    if not consistency_ok:
        return "不成立"
    values = [float(v) for v in speedups if v is not None]
    if not values:
        return "不成立"
    return "通过" if max(values) > 1.0 else "负结果"


@dataclass
class BenchReport:
    """一次评测的完整报告。"""

    config: dict
    env: dict
    readings: dict
    account: dict
    tasks: dict
    conclusion: str

    def as_dict(self) -> dict:
        return {
            "config": self.config,
            "env": self.env,
            "readings": self.readings,
            "account": self.account,
            "tasks": self.tasks,
            "conclusion": self.conclusion,
        }

    def save_json(self, path: str | Path) -> None:
        import json

        text = json.dumps(
            self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True
        )
        Path(path).write_text(text, encoding="utf-8")

    def plot(self, path: str | Path) -> None:
        """权衡曲线:横轴接受率,纵轴实测加速比,每个点标出 gamma。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        runs = sorted(
            self.readings["R6"]["runs"].values(), key=lambda run: run["gamma"]
        )
        xs = [float(run["stats"]["accept_rate"]) for run in runs]
        ys = [float(run["speedup"]) for run in runs]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(xs, ys, marker="o")
        for run, x, y in zip(runs, xs, ys):
            ax.annotate(
                f"g={run['gamma']}", (x, y), textcoords="offset points", xytext=(4, 4)
            )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("accept rate")
        ax.set_ylabel("speedup (x)")
        ax.set_title("Every N-gram: speedup vs accept rate")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def run_fusion_scan(
    model,
    memory,
    *,
    prompt_ids,
    cases,
    ood_ids,
    lambdas,
    modes=("arithmetic",),
    gamma: int = 4,
    max_new_tokens: int = 32,
    repeats: int = 3,
) -> dict:
    """扫 mode × λ,每个配置产出 (接受率, 加速比, 域内命中率, 域外困惑度)。"""
    from everyngram.decode import greedy_generate
    from everyngram.fuse import FusionSpec
    from everyngram.tasks import next_token_accuracy, perplexity

    baseline_timing = measure_latency(
        lambda: greedy_generate(
            model, prompt_ids, max_new_tokens=max_new_tokens, use_cache=True
        ),
        repeats=repeats,
    )
    baseline_ms = baseline_timing["median_s"] * 1000.0 / max(max_new_tokens, 1)
    baseline = {
        "ms_per_token": round(baseline_ms, 4),
        "in_domain": round(next_token_accuracy(model, cases), 4),
        "ood_perplexity": round(perplexity(model, ood_ids), 4),
    }

    rows: list[dict] = []
    for mode in modes:
        for lam in lambdas:
            spec = FusionSpec(lam=float(lam), mode=mode, memory=memory)
            timing = measure_latency(
                lambda s=spec: greedy_generate(
                    model,
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    memory=memory,
                    gamma=gamma,
                    fusion=s,
                    use_cache=True,
                ),
                repeats=repeats,
            )
            _, stats = greedy_generate(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                memory=memory,
                gamma=gamma,
                fusion=spec,
                use_cache=True,
            )
            ms_per_token = timing["median_s"] * 1000.0 / max(max_new_tokens, 1)
            rows.append(
                {
                    "mode": mode,
                    "lam": float(lam),
                    "gamma": int(gamma),
                    "accept_rate": stats.as_dict()["accept_rate"],
                    "accept_length": stats.as_dict()["accept_length"],
                    "ms_per_token": round(ms_per_token, 4),
                    "speedup": (
                        round(baseline_ms / ms_per_token, 4) if ms_per_token else None
                    ),
                    "in_domain": round(
                        next_token_accuracy(model, cases, fusion=spec), 4
                    ),
                    "ood_perplexity": round(perplexity(model, ood_ids, fusion=spec), 4),
                }
            )

    return {"baseline": baseline, "rows": rows}


def plot_tradeoff(scan: dict, path: str | Path) -> None:
    """权衡曲线:左图散点(质量对接受率,标 λ),右图以 λ 为横轴的加速比。

    横轴接受率是**观测量**而不是控制量,对 λ 不单调,连线会造出并不存在的
    过渡段,因此左图只画散点并逐点标 λ。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_q, ax_s) = plt.subplots(1, 2, figsize=(10, 4))
    modes = sorted({row["mode"] for row in scan["rows"]})
    for mode in modes:
        rows = sorted(
            (row for row in scan["rows"] if row["mode"] == mode),
            key=lambda row: row["lam"],
        )
        xs = [row["accept_rate"] for row in rows]
        ys = [row["in_domain"] for row in rows]
        ax_q.scatter(xs, ys, label=mode)
        for row, x, y in zip(rows, xs, ys):
            ax_q.annotate(
                f"λ={row['lam']}", (x, y), textcoords="offset points", xytext=(4, 4)
            )
        ax_s.plot(
            [row["lam"] for row in rows],
            [row["speedup"] for row in rows],
            marker="o",
            label=mode,
        )

    baseline = scan.get("baseline", {})
    if "in_domain" in baseline:
        ax_q.axhline(baseline["in_domain"], color="gray", linestyle="--", linewidth=1)
    ax_s.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    for ax in (ax_q, ax_s):
        ax.legend()
        ax.grid(alpha=0.3)
    ax_q.set_xlabel("accept rate")
    ax_q.set_ylabel("in-domain accuracy")
    ax_q.set_title("quality vs accept rate (scatter, λ labeled)")
    ax_s.set_xlabel("λ (fusion strength)")
    ax_s.set_ylabel("speedup (x)")
    ax_s.set_title("speed vs λ")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_budget_selection(report: dict, path: str | Path) -> None:
    """增量 9:预算选择的帕累托散点(质量 vs 接受覆盖)+ 命中率柱状图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = list(report.get("rows", []))
    rows = [row for row in all_rows if row.get("strategy") != "full"]
    full_rows = [row for row in all_rows if row.get("strategy") == "full"]
    budgets = sorted({int(row["entries"]) for row in rows})
    strategies = sorted({str(row["strategy"]) for row in rows})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(budgets), 1)))
    for color, entries in zip(colors, budgets):
        subset = [row for row in rows if int(row["entries"]) == entries]
        for row in subset:
            if row.get("accept_coverage") is None or row.get("mean") is None:
                continue
            ax.scatter(
                float(row["accept_coverage"]),
                float(row["mean"]),
                color=color,
                s=44,
                zorder=3,
            )
            ax.annotate(
                str(row["strategy"]),
                (float(row["accept_coverage"]), float(row["mean"])),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.scatter([], [], color=color, label=f"B={entries}")
    for row in full_rows:
        ax.axhline(
            float(row["mean"]),
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"full table (G={float(row['mean']):+.4f})",
        )
    ax.set_xlabel("draft accept coverage (accepted drafts / positions)")
    ax.set_ylabel("quality gain G (nats, file-clustered)")
    ax.set_title("quality vs draft acceptance at fixed budgets")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.8 / max(len(budgets), 1)
    x = np.arange(len(strategies))
    for i, entries in enumerate(budgets):
        subset = {str(row["strategy"]): row for row in rows if int(row["entries"]) == entries}
        heights = [float(subset.get(s, {}).get("hit_rate") or 0.0) for s in strategies]
        ax.bar(x + i * width, heights, width, label=f"B={entries}")
    for row in full_rows:
        ax.axhline(
            float(row.get("hit_rate") or 0.0),
            color="black",
            linestyle="--",
            linewidth=1,
            label="full table",
        )
    ax.set_xticks(x + width * (len(budgets) - 1) / 2 if budgets else x)
    ax.set_xticklabels(strategies, fontsize=8)
    ax.set_ylabel("memory hit rate")
    ax.set_title("hit rate by strategy")
    ax.grid(alpha=0.25, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gated_experiment(result: dict, path: str | Path) -> None:
    """门控实验:test 上各配置的配对 NLL 增益(带标准误)对接受率。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, block in result["test"]["configs"].items():
        delta = block["nll_delta"]
        x = block.get("accept_rate")
        if x is None:
            continue
        label = f"{name} (λ={block['lam']}"
        if block.get("gate"):
            label += f", gate={block['gate']})"
        else:
            label += ", no gate)"
        ax.errorbar(
            [x], [delta["mean"]], yerr=[delta["se"]], marker="o", capsize=4, label=label
        )
        ax.annotate(
            f"EM={block['em']:.2f}",
            (x, delta["mean"]),
            textcoords="offset points",
            xytext=(6, -12),
        )
    best = result.get("selection", {}).get("best")
    if best:
        ax.set_title(
            f"dev picked λ={best['lam']}, gate coverage {best['coverage']:.2f}\n"
            f"(order≥{best['gate']['min_order']}, share≥{best['gate']['min_share']}, "
            f"top_prob≤{best['gate']['max_top_prob']})",
            fontsize=10,
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("accept rate")
    ax.set_ylabel("paired quality gain G (NLL, mean ± SE)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_budget_scan(
    model,
    token_ids,
    *,
    fingerprint: str,
    budgets,
    cases,
    lambdas=(0.0, 0.2),
    gamma: int = 4,
    max_new_tokens: int = 32,
    repeats: int = 3,
    out_dir: str | Path | None = None,
) -> dict:
    """Q1:同一语料扫预算 B,每个 B 测 (TV, NLL 增益, 接受率, 加速比)。"""
    from everyngram.decode import greedy_generate
    from everyngram.fuse import FusionSpec
    from everyngram.memory import Memory
    from everyngram.tasks import next_token_nll, tv_distance

    prompt_ids = list(cases[0].prompt_ids) if cases else [0]
    baseline_timing = measure_latency(
        lambda: greedy_generate(
            model, prompt_ids, max_new_tokens=max_new_tokens, use_cache=True
        ),
        repeats=repeats,
    )
    baseline_ms = baseline_timing["median_s"] * 1000.0 / max(max_new_tokens, 1)
    baseline_nll = next_token_nll(model, cases)
    baseline = {
        "ms_per_token": round(baseline_ms, 4),
        "nll": round(baseline_nll, 4),
        "cases": len(cases),
    }

    rows: list[dict] = []
    for budget in budgets:
        memory = Memory.build(
            list(token_ids),
            fingerprint=fingerprint,
            orders=(1, 2, 3),
            budget=int(budget),
            continuation_len=16,
        )
        file_bytes = None
        if out_dir is not None:
            path = Path(out_dir) / f"inc4_b{int(budget)}.ngmem"
            path.parent.mkdir(parents=True, exist_ok=True)
            memory.save(path)
            file_bytes = path.stat().st_size
        tv = tv_distance(model, cases, memory)
        for lam in lambdas:
            spec = FusionSpec(lam=float(lam), mode="arithmetic", memory=memory)
            timing = measure_latency(
                lambda s=spec, m=memory: greedy_generate(
                    model,
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    memory=m,
                    gamma=gamma,
                    fusion=s,
                    use_cache=True,
                ),
                repeats=repeats,
            )
            _, stats = greedy_generate(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                memory=memory,
                gamma=gamma,
                fusion=spec,
                use_cache=True,
            )
            ms_per_token = timing["median_s"] * 1000.0 / max(max_new_tokens, 1)
            nll = next_token_nll(model, cases, fusion=spec)
            stats_dict = stats.as_dict()
            rows.append(
                {
                    "budget": int(budget),
                    "entries": memory.stats()["entries"],
                    "dropped": memory.stats()["dropped"],
                    "file_bytes": file_bytes,
                    "lam": float(lam),
                    "gamma": int(gamma),
                    "tv_mean": (
                        None if tv["tv_mean"] is None else round(tv["tv_mean"], 4)
                    ),
                    "hit_rate": round(tv["hit_rate"], 4),
                    "accept_rate": stats_dict["accept_rate"],
                    "accept_length": stats_dict["accept_length"],
                    "ms_per_token": round(ms_per_token, 4),
                    "speedup": (
                        round(baseline_ms / ms_per_token, 4) if ms_per_token else None
                    ),
                    "nll": round(nll, 4),
                    "nll_gain": round(baseline_nll - nll, 4),
                }
            )

    return {"baseline": baseline, "rows": rows}


def _memory_from_pool(source_memory, picked, pool_name: str):
    """用打分结果里的 key 反查全量表条目,构造一个只含该池的记忆表。"""
    from everyngram.memory import Memory

    entries = {}
    for item in picked:
        key = tuple(item["key"])
        entry = source_memory.entries.get(key)
        if entry is not None:
            entries[key] = entry
    if not entries:
        raise ValueError(f"池 {pool_name} 为空:打分结果里的 key 都不在全量表中")
    return Memory(
        fingerprint=source_memory.fingerprint,
        orders=source_memory.orders,
        budget=len(entries),
        continuation_len=source_memory.continuation_len,
        next_top_k=source_memory.next_top_k,
        entries=entries,
    )


def run_pool_experiment(
    model,
    memory,
    pools: dict,
    *,
    cases,
    gamma: int = 4,
    lambdas=(0.0, 0.2),
    max_new_tokens: int = 32,
    repeats: int = 3,
) -> dict:
    """C1:等预算的加速池与质量池,在同一批用例上比 (接受率, NLL 增益)。"""
    from everyngram.decode import greedy_generate
    from everyngram.fuse import FusionSpec
    from everyngram.tasks import next_token_nll, tv_distance

    picked = {
        "speed_pool": pools["speed_pool"],
        "quality_pool": pools["quality_pool"],
        "union_pool": list(pools["speed_pool"]) + list(pools["quality_pool"]),
    }
    prompt_ids = list(cases[0].prompt_ids) if cases else [0]
    baseline_nll = next_token_nll(model, cases)
    baseline_timing = measure_latency(
        lambda: greedy_generate(
            model, prompt_ids, max_new_tokens=max_new_tokens, use_cache=True
        ),
        repeats=repeats,
    )
    baseline_ms = baseline_timing["median_s"] * 1000.0 / max(max_new_tokens, 1)

    blocks: dict[str, dict] = {}
    for name, items in picked.items():
        pool_memory = _memory_from_pool(memory, items, name)
        tv = tv_distance(model, cases, pool_memory)
        rows = []
        for lam in lambdas:
            spec = FusionSpec(lam=float(lam), mode="arithmetic", memory=pool_memory)
            timing = measure_latency(
                lambda s=spec, m=pool_memory: greedy_generate(
                    model,
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    memory=m,
                    gamma=gamma,
                    fusion=s,
                    use_cache=True,
                ),
                repeats=repeats,
            )
            _, stats = greedy_generate(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                memory=pool_memory,
                gamma=gamma,
                fusion=spec,
                use_cache=True,
            )
            ms_per_token = timing["median_s"] * 1000.0 / max(max_new_tokens, 1)
            nll = next_token_nll(model, cases, fusion=spec)
            stats_dict = stats.as_dict()
            rows.append(
                {
                    "lam": float(lam),
                    "accept_rate": stats_dict["accept_rate"],
                    "accept_length": stats_dict["accept_length"],
                    "ms_per_token": round(ms_per_token, 4),
                    "speedup": (
                        round(baseline_ms / ms_per_token, 4) if ms_per_token else None
                    ),
                    "nll": round(nll, 4),
                    "nll_gain": round(baseline_nll - nll, 4),
                }
            )
        blocks[name] = {
            "entries": pool_memory.stats()["entries"],
            "tv_mean": None if tv["tv_mean"] is None else round(tv["tv_mean"], 4),
            "hit_rate": round(tv["hit_rate"], 4),
            "rows": rows,
        }

    return {
        "baseline": {
            "nll": round(baseline_nll, 4),
            "ms_per_token": round(baseline_ms, 4),
        },
        "overlap": pools["overlap"],
        "k_effective": pools["k_effective"],
        "pools": blocks,
    }


def plot_budget_scan(scan: dict, path: str | Path) -> None:
    """Q1:上排 G 与 S 对 B,下排 G×S 对 B。B 是单调控制变量,折线在这里合法。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in scan["rows"] if float(row["lam"]) > 0.0]
    lams = sorted({row["lam"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for lam in lams:
        series = sorted(
            (row for row in rows if row["lam"] == lam), key=lambda row: row["budget"]
        )
        xs = [row["budget"] for row in series]
        axes[0].plot(xs, [row["nll_gain"] for row in series], marker="o", label=f"λ={lam}")
        axes[1].plot(xs, [row["speedup"] for row in series], marker="o", label=f"λ={lam}")
        axes[2].plot(
            xs,
            [row["nll_gain"] * (row["speedup"] or 0.0) for row in series],
            marker="o",
            label=f"λ={lam}",
        )
    axes[0].set_ylabel("quality gain G (NLL)")
    axes[1].set_ylabel("speedup S (x)")
    axes[2].set_ylabel("G x S")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[2].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    for ax, title in zip(
        axes, ("quality vs budget", "speed vs budget", "product vs budget")
    ):
        ax.set_xlabel("budget B (entries)")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pool_experiment(result: dict, path: str | Path) -> None:
    """C1:左图是各池在 (接受率, NLL 增益) 平面的位置,右图是交叠率随 k 的曲线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = result.get("overlap_curve")
    if curve:
        fig, (ax, ax_k) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax_k = None
    for name, block in result["pools"].items():
        rows = [row for row in block["rows"] if float(row["lam"]) > 0.0]
        if not rows:
            continue
        xs = [row["accept_rate"] for row in rows]
        ys = [row["nll_gain"] for row in rows]
        ax.plot(xs, ys, marker="o", label=f"{name} ({block['entries']} entries)")
        for row, x, y in zip(rows, xs, ys):
            ax.annotate(
                f"λ={row['lam']}", (x, y), textcoords="offset points", xytext=(4, 4)
            )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("accept rate")
    ax.set_ylabel("quality gain G (NLL)")
    ax.set_title(f"C1: speed pool vs quality pool (overlap {result['overlap']:.2f})")
    ax.grid(alpha=0.3)
    ax.legend()
    if ax_k is not None:
        ax_k.plot(curve["ks"], curve["overlap"], marker="o")
        ax_k.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax_k.set_xlabel("k (pool size)")
        ax_k.set_ylabel("Jaccard overlap")
        ax_k.set_title("overlap vs pool size")
        ax_k.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_gated_experiment(
    model,
    memory,
    *,
    dev_cases,
    test_cases,
    gates,
    lambdas,
    min_coverage: float = 0.25,
    gamma: int = 4,
    em_tokens: int = 8,
    timing_tokens: int = 32,
    repeats: int = 3,
) -> dict:
    """dev 选门槛与 λ,冻结后在 test 上评一次,并与无门槛配置对照。"""
    from everyngram.decode import greedy_generate
    from everyngram.fuse import FusionSpec
    from everyngram.select import select_gate
    from everyngram.tasks import (
        completion_exact_match,
        next_token_accuracy,
        next_token_nll,
        paired_nll_delta,
    )

    selection = select_gate(
        model, memory, dev_cases, gates=gates, lambdas=lambdas,
        min_coverage=min_coverage,
    )
    best = selection["best"]
    if best is not None:
        from everyngram.fuse import Gate

        gated_spec = FusionSpec(
            lam=float(best["lam"]), mode="arithmetic", memory=memory,
            gate=Gate(**best["gate"]),
        )
        shared_lam = float(best["lam"])
    else:
        gated_spec = None
        shared_lam = float(lambdas[0])

    configs: dict[str, FusionSpec] = {
        "ungated_same_lam": FusionSpec(lam=shared_lam, mode="arithmetic", memory=memory),
        "ungated_lam_0.2": FusionSpec(lam=0.2, mode="arithmetic", memory=memory),
    }
    if gated_spec is not None:
        configs = {"gated": gated_spec, **configs}

    prompt_ids = list(test_cases[0].prompt_ids) if test_cases else [0]
    baseline_timing = measure_latency(
        lambda: greedy_generate(
            model, prompt_ids, max_new_tokens=timing_tokens, use_cache=True
        ),
        repeats=repeats,
    )
    baseline_ms = baseline_timing["median_s"] * 1000.0 / max(timing_tokens, 1)
    baseline = {
        "nll": round(next_token_nll(model, test_cases), 4),
        "accuracy": round(next_token_accuracy(model, test_cases), 4),
        "em": round(completion_exact_match(model, test_cases, max_new_tokens=em_tokens), 4),
        "ms_per_token": round(baseline_ms, 4),
        "n_cases": len(test_cases),
    }

    blocks: dict[str, dict] = {}
    for name, spec in configs.items():
        timing = measure_latency(
            lambda s=spec: greedy_generate(
                model,
                prompt_ids,
                max_new_tokens=timing_tokens,
                memory=memory,
                gamma=gamma,
                fusion=s,
                use_cache=True,
            ),
            repeats=repeats,
        )
        _, stats = greedy_generate(
            model,
            prompt_ids,
            max_new_tokens=timing_tokens,
            memory=memory,
            gamma=gamma,
            fusion=spec,
            use_cache=True,
        )
        ms_per_token = timing["median_s"] * 1000.0 / max(timing_tokens, 1)
        delta = paired_nll_delta(model, test_cases, fusion=spec)
        stats_dict = stats.as_dict()
        blocks[name] = {
            "gate": None if spec.gate is None else spec.gate.as_dict(),
            "lam": float(spec.lam),
            "nll_delta": {
                "n": delta["n"],
                "mean": round(delta["mean"], 4),
                "se": round(delta["se"], 4),
                "improved": delta["improved"],
            },
            "accuracy": round(next_token_accuracy(model, test_cases, fusion=spec), 4),
            "em": round(
                completion_exact_match(
                    model, test_cases, fusion=spec, max_new_tokens=em_tokens
                ),
                4,
            ),
            "accept_rate": stats_dict["accept_rate"],
            "accept_length": stats_dict["accept_length"],
            "ms_per_token": round(ms_per_token, 4),
            "speedup": round(baseline_ms / ms_per_token, 4) if ms_per_token else None,
        }

    return {
        "selection": {
            "best": best,
            "n_positions": selection["n_positions"],
            "rows": selection["rows"],
        },
        "test": {"baseline": baseline, "configs": blocks},
    }


def make_folds(
    files: Sequence[str | Path], *, k: int = 4, seed: int = 0
) -> tuple[dict, ...]:
    """把文件切成 k 折:第 i 折的 test 取 files[i::k],dev 取 files[(i+1)%k::k]。

    每个文件恰好当一次 test;每折的 test 与 dev 都不同,便于"dev 选参、
    test 评一次"在多折上重复(审计指出单次切分的结论不可迁移)。
    """
    import random

    items = [str(path) for path in files]
    k = int(k)
    if k < 3:
        # stride 切分下 test 与 dev 各占 n/k 个文件,k < 3 时建表语料会为空
        raise ValueError(f"折数必须 >= 3: {k}")
    if len(items) < 3 * k:
        raise ValueError(
            f"文件数不足:需要至少 {3 * k} 个(建表/dev/test 各 k),实际 {len(items)}"
        )
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    folds = []
    for i in range(k):
        test = shuffled[i::k]
        dev = shuffled[(i + 1) % k :: k]
        held = set(test) | set(dev)
        corpus = [path for path in shuffled if path not in held]
        if not corpus:
            raise ValueError(
                f"文件数不足以支撑 {k} 折:第 {i} 折的建表语料为空"
                f"(test={len(test)}, dev={len(dev)})"
            )
        folds.append({"test": test, "dev": dev, "corpus": corpus})
    return tuple(folds)


def run_fold_experiment(
    model,
    tokenizer,
    *,
    folds,
    fingerprint: str,
    gates,
    lambdas,
    dev_cases: int = 24,
    test_cases: int = 24,
    corpus_orders=(1, 2, 3),
    min_coverage: float = 0.25,
    gamma: int = 4,
    em_tokens: int = 1,
    prefix_tokens: int = 8,
    timing_tokens: int = 32,
    repeats: int = 1,
) -> dict:
    """每折独立建表、dev 选参、冻结后在 test 上评一次,再给出折间汇总。

    汇总里 `mean_across_folds` 与 `se_across_folds` 是最终判据:单折的
    dev→test 迁移在增量 5 已经被证伪,只有折间分布才算证据。
    """
    from everyngram.decode import greedy_generate
    from everyngram.fuse import FusionSpec, Gate
    from everyngram.memory import Memory
    from everyngram.select import select_gate
    from everyngram.tasks import (
        build_completion_task,
        build_corpus_ids,
        clustered_delta_stats,
        completion_exact_match,
        completion_prefix_score,
        next_token_nll,
        paired_nll_delta,
        split_files,
    )

    fold_rows: list[dict] = []
    for index, fold in enumerate(folds):
        corpus_ids = build_corpus_ids(fold["corpus"], tokenizer)
        memory = Memory.build(
            corpus_ids, fingerprint=fingerprint, orders=tuple(corpus_orders),
            budget=1_000_000,
        )
        dev_task = build_completion_task(fold["dev"], tokenizer, n_cases=dev_cases)
        test_task = build_completion_task(fold["test"], tokenizer, n_cases=test_cases)
        row: dict = {
            "fold": index,
            "corpus_files": len(fold["corpus"]),
            "dev_files": len(fold["dev"]),
            "test_files": len(fold["test"]),
            "corpus_tokens": len(corpus_ids),
            "entries": memory.stats()["entries"],
            "n_dev_cases": len(dev_task.cases),
            "n_test_cases": len(test_task.cases),
        }
        if not test_task.cases:
            row["skipped"] = "test 用例为 0"
            fold_rows.append(row)
            continue

        selection = select_gate(
            model, memory, dev_task.cases, gates=gates, lambdas=lambdas,
            min_coverage=min_coverage,
        )
        best = selection["best"]
        row["dev_best"] = best
        if best is None:
            row["skipped"] = "dev 无配置达标"
            fold_rows.append(row)
            continue

        gated = FusionSpec(
            lam=float(best["lam"]), mode="arithmetic", memory=memory,
            gate=Gate(**best["gate"]),
        )
        ungated = FusionSpec(lam=float(best["lam"]), mode="arithmetic", memory=memory)
        prompt_ids = list(test_task.cases[0].prompt_ids)
        baseline_timing = measure_latency(
            lambda: greedy_generate(
                model, prompt_ids, max_new_tokens=timing_tokens, use_cache=True
            ),
            repeats=repeats,
        )
        baseline_ms = baseline_timing["median_s"] * 1000.0 / max(timing_tokens, 1)

        configs: dict[str, dict] = {}
        for name, spec in (("gated", gated), ("ungated_same_lam", ungated)):
            delta = paired_nll_delta(model, test_task.cases, fusion=spec)
            pairs = [
                (case.origin or f"case-{i}", value)
                for i, (case, value) in enumerate(
                    zip(test_task.cases, delta["deltas"])
                )
            ]
            timing = measure_latency(
                lambda s=spec: greedy_generate(
                    model, prompt_ids, max_new_tokens=timing_tokens,
                    memory=memory, gamma=gamma, fusion=s, use_cache=True,
                ),
                repeats=repeats,
            )
            _, stats = greedy_generate(
                model, prompt_ids, max_new_tokens=timing_tokens,
                memory=memory, gamma=gamma, fusion=spec, use_cache=True,
            )
            ms_per_token = timing["median_s"] * 1000.0 / max(timing_tokens, 1)
            configs[name] = {
                "lam": float(spec.lam),
                "gate": None if spec.gate is None else spec.gate.as_dict(),
                "clustered": clustered_delta_stats(pairs),
                "nll": round(next_token_nll(model, test_task.cases, fusion=spec), 4),
                "prefix": round(
                    completion_prefix_score(
                        model, test_task.cases, fusion=spec, max_new_tokens=prefix_tokens
                    ),
                    4,
                ),
                "em1": round(
                    completion_exact_match(
                        model, test_task.cases, fusion=spec, max_new_tokens=em_tokens
                    ),
                    4,
                ),
                "accept_rate": stats.as_dict()["accept_rate"],
                "accept_length": stats.as_dict()["accept_length"],
                "speedup": round(baseline_ms / ms_per_token, 4) if ms_per_token else None,
            }
        row["test"] = {
            "baseline": {
                "nll": round(next_token_nll(model, test_task.cases), 4),
                "prefix": round(
                    completion_prefix_score(
                        model, test_task.cases, max_new_tokens=prefix_tokens
                    ),
                    4,
                ),
                "em1": round(
                    completion_exact_match(
                        model, test_task.cases, max_new_tokens=em_tokens
                    ),
                    4,
                ),
                "ms_per_token": round(baseline_ms, 4),
            },
            "configs": configs,
        }
        fold_rows.append(row)

    pooled: dict[str, dict] = {}
    for name in ("gated", "ungated_same_lam"):
        deltas = [
            row["test"]["configs"][name]["clustered"]["mean"]
            for row in fold_rows
            if "test" in row and name in row["test"]["configs"]
        ]
        if not deltas:
            continue
        import numpy as np

        pooled[name] = {
            "n_folds": len(deltas),
            "mean_across_folds": round(float(np.mean(deltas)), 4),
            "se_across_folds": (
                round(float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))), 4)
                if len(deltas) > 1
                else 0.0
            ),
            "per_fold_delta": [round(value, 4) for value in deltas],
            "per_fold_improved": int(sum(1 for value in deltas if value > 0)),
        }

    return {"folds": fold_rows, "pooled": pooled}


def plot_fold_experiment(report: dict, path: str | Path) -> None:
    """逐折 gated 配对增益(文件级聚类)与折间均值 ± 折间标准误,按语料分色。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, block in report.get("corpora", {}).items():
        pooled = (block or {}).get("pooled", {}).get("gated")
        if not pooled:
            continue
        deltas = pooled["per_fold_delta"]
        ax.plot(
            range(1, len(deltas) + 1),
            deltas,
            marker="o",
            label=f"{name} (mean {pooled['mean_across_folds']:+.4f} "
            f"± {pooled['se_across_folds']:.4f}, {pooled['n_folds']} folds)",
        )
        ax.axhline(pooled["mean_across_folds"], linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("fold")
    ax.set_ylabel("gated paired quality gain G (NLL, file-clustered)")
    ax.set_title("per-fold gated gain (increment 6)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_power_experiment(
    model,
    tokenizer,
    *,
    corpus_root,
    fingerprint: str,
    gates,
    lambdas,
    exclude=(),
    memory_files: int = 40,
    dev_files: int = 20,
    test_files: int = 240,
    pool_files: int | None = None,
    memory_offset: int = 0,
    dev_cases_per_file: int = 12,
    test_cases_per_file: int = 4,
    corpus_orders=(1, 2, 3),
    min_coverage: float = 0.25,
    em_tokens: int = 1,
    bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """大样本判定:记忆表建一次,dev 选参冻结,在大量 test 文件上评一次。

    增量 6 的折间离散来自文件异质性(σ_file ≈ 0.082),所以压标准误的杠杆是
    **测试文件数**而不是折数:均值的 SE ≈ σ_file / sqrt(文件数)。
    """
    import random

    from everyngram.fuse import FusionSpec, Gate
    from everyngram.memory import Memory
    from everyngram.select import select_gate
    from everyngram.tasks import (
        build_completion_task,
        build_corpus_ids,
        clustered_delta_stats,
        collect_code_files,
        completion_exact_match,
        next_token_accuracy,
        next_token_nll,
        paired_nll_delta,
    )

    pool_size = int(pool_files) if pool_files is not None else int(memory_files)
    if pool_size < int(memory_files):
        raise ValueError(f"pool_files({pool_size}) 不能小于 memory_files({memory_files})")
    needed = pool_size + int(dev_files) + int(test_files)
    files = collect_code_files(
        [corpus_root],
        max_files=needed,
        seed=seed,
        exclude_substrings=tuple(exclude),
    )
    if len(files) < needed:
        raise ValueError(
            f"可用文件不足:需要 {needed} 个"
            f"(memory {memory_files} + dev {dev_files} + test {test_files}),"
            f"实际 {len(files)}"
        )
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    memory_pool = shuffled[:pool_size]
    dev_set = shuffled[pool_size : pool_size + int(dev_files)]
    test_set = shuffled[pool_size + int(dev_files) : needed]
    if int(memory_offset) and memory_pool:
        offset = int(memory_offset) % len(memory_pool)
        memory_pool = memory_pool[offset:] + memory_pool[:offset]
    memory_set = memory_pool[: int(memory_files)]

    corpus_ids = build_corpus_ids(memory_set, tokenizer)
    memory = Memory.build(
        corpus_ids, fingerprint=fingerprint, orders=tuple(corpus_orders), budget=1_000_000
    )
    dev_task = build_completion_task(
        dev_set, tokenizer, n_cases=int(dev_files) * int(dev_cases_per_file)
    )
    test_task = build_completion_task(
        test_set, tokenizer, n_cases=int(test_files) * int(test_cases_per_file)
    )

    selection = select_gate(
        model, memory, dev_task.cases, gates=gates, lambdas=lambdas,
        min_coverage=min_coverage,
    )
    best = selection["best"]
    result: dict = {
        "counts": {
            "memory": len(memory_set),
            "memory_pool": pool_size,
            "dev": len(dev_set),
            "test": len(test_set),
            "corpus_tokens": len(corpus_ids),
            "entries": memory.stats()["entries"],
        },
        "dev": {
            "n_cases": len(dev_task.cases),
            "n_files": len({case.origin for case in dev_task.cases}),
            "coverage_floor": float(min_coverage),
        },
        "split": {
            "memory": [Path(path).name for path in memory_set],
            "dev": [Path(path).name for path in dev_set],
            "test": [Path(path).name for path in test_set],
        },
        "selected": best,
    }
    if best is None:
        result["skipped"] = "dev 无配置达标"
        return result

    gated = FusionSpec(
        lam=float(best["lam"]), mode="arithmetic", memory=memory,
        gate=Gate(**best["gate"]),
    )
    ungated = FusionSpec(lam=float(best["lam"]), mode="arithmetic", memory=memory)

    baseline = {
        "nll": round(next_token_nll(model, test_task.cases), 4),
        "accuracy": round(next_token_accuracy(model, test_task.cases), 4),
        "em1": round(
            completion_exact_match(
                model, test_task.cases, max_new_tokens=int(em_tokens)
            ),
            4,
        ),
        "n_cases": len(test_task.cases),
    }

    configs: dict[str, dict] = {}
    for name, spec in (("gated", gated), ("ungated_same_lam", ungated)):
        delta = paired_nll_delta(model, test_task.cases, fusion=spec)
        pairs = [
            (case.origin or f"case-{i}", value)
            for i, (case, value) in enumerate(zip(test_task.cases, delta["deltas"]))
        ]
        stats = clustered_delta_stats(pairs, bootstrap=bootstrap, seed=seed)
        per_file: dict[str, float] = {}
        for group, value in pairs:
            per_file.setdefault(str(group), []).append(float(value))
        configs[name] = {
            "lam": float(spec.lam),
            "gate": None if spec.gate is None else spec.gate.as_dict(),
            "mean": round(stats["mean"], 6),
            "se": round(stats["se"], 6),
            "ci_low": None if stats["ci_low"] is None else round(stats["ci_low"], 6),
            "ci_high": None if stats["ci_high"] is None else round(stats["ci_high"], 6),
            "improved_groups": stats["improved_groups"],
            "n_test_files": stats["n_groups"],
            "n_test_cases": stats["n_cases"],
            "clustered": stats,
            "per_file_deltas": {
                group: round(float(np.mean(values)), 6)
                for group, values in sorted(per_file.items())
            },
            "nll": round(next_token_nll(model, test_task.cases, fusion=spec), 4),
        }
    configs["gated"]["em1"] = round(
        completion_exact_match(
            model, test_task.cases, fusion=gated, max_new_tokens=int(em_tokens)
        ),
        4,
    )
    result["test"] = {"baseline": baseline, **configs}
    result["test"]["gated"]["em1_baseline"] = baseline["em1"]
    return result


def run_budget_selection_experiment(
    model,
    tokenizer,
    *,
    corpus_root,
    fingerprint: str,
    gates,
    lambdas,
    exclude=(),
    pool_files: int = 40,
    dev_files: int = 20,
    test_files: int = 240,
    dev_cases_per_file: int = 12,
    test_cases_per_file: int = 2,
    corpus_orders=(1, 2, 3),
    min_coverage: float = 0.25,
    budgets=(500, 2000, 8000),
    strategies=("freq", "query", "quality", "speed", "hybrid"),
    bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """增量 9:固定预算下的内容选择,单次前向复用评所有策略。

    切分 → 全表 → dev 选 λ/门槛并冻结 → dev 条目打分 → 每档预算建子表 →
    test 上每个用例只做一次前向,在同一份 logits 上评所有表(质量、命中、接受)。
    """
    import hashlib
    import random

    from everyngram.budget import STRATEGIES, score_entry_usage, select_keys
    from everyngram.fuse import Gate, fuse_probs, memory_dist, softmax
    from everyngram.memory import Memory
    from everyngram.select import select_gate
    from everyngram.tasks import (
        build_completion_task,
        build_corpus_ids,
        clustered_delta_stats,
        collect_code_files,
    )

    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise ValueError(f"未知策略 {strategy}: 必须是 {STRATEGIES} 之一")

    needed = int(pool_files) + int(dev_files) + int(test_files)
    files = collect_code_files(
        [corpus_root],
        max_files=needed,
        seed=seed,
        exclude_substrings=tuple(exclude),
    )
    if len(files) < needed:
        raise ValueError(
            f"可用文件不足:需要 {needed} 个(pool {pool_files} + dev {dev_files}"
            f" + test {test_files}),实际 {len(files)}"
        )
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    memory_set = shuffled[: int(pool_files)]
    dev_set = shuffled[int(pool_files) : int(pool_files) + int(dev_files)]
    test_set = shuffled[int(pool_files) + int(dev_files) : needed]

    corpus_ids = build_corpus_ids(memory_set, tokenizer)
    memory = Memory.build(
        corpus_ids,
        fingerprint=fingerprint,
        orders=tuple(corpus_orders),
        budget=1_000_000,
    )
    dev_task = build_completion_task(
        dev_set, tokenizer, n_cases=int(dev_files) * int(dev_cases_per_file)
    )
    test_task = build_completion_task(
        test_set, tokenizer, n_cases=int(test_files) * int(test_cases_per_file)
    )

    selection = select_gate(
        model,
        memory,
        dev_task.cases,
        gates=gates,
        lambdas=lambdas,
        min_coverage=min_coverage,
    )
    best = selection["best"]
    n_entries = len(memory.entries)
    result: dict = {
        "counts": {
            "memory": len(memory_set),
            "dev": len(dev_set),
            "test": len(test_set),
            "corpus_tokens": len(corpus_ids),
            "entries": n_entries,
        },
        "split": {
            "memory": [Path(path).name for path in memory_set],
            "dev": [Path(path).name for path in dev_set],
            "test": [Path(path).name for path in test_set],
        },
        "split_paths": {
            "memory": [str(path) for path in memory_set],
            "dev": [str(path) for path in dev_set],
            "test": [str(path) for path in test_set],
        },
        "dev": {
            "n_cases": len(dev_task.cases),
            "coverage_floor": float(min_coverage),
        },
        "dev_selection_rows": selection["rows"],
        "selected": best,
    }
    if best is None:
        result["skipped"] = "dev 无配置达标"
        return result

    gate = Gate(**best["gate"])
    lam = float(best["lam"])
    usage = score_entry_usage(model, memory, dev_task.cases, lam=lam, gate=gate)
    result["dev_usage"] = {
        "positions": usage["positions"],
        "hits": usage["hits"],
        "keys_used": len(usage["per_key"]),
        "keys_total": n_entries,
    }

    budget_values: list[int] = []
    for raw in budgets:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"预算必须为正整数: {raw}")
        capped = min(value, n_entries)
        if capped not in budget_values:
            budget_values.append(capped)
    budget_values.sort()

    tables: dict[str, dict] = {}
    for entries_b in budget_values:
        for strategy in strategies:
            keys = select_keys(memory, usage, strategy=strategy, budget=entries_b)
            row_id = f"{strategy}@{entries_b}"
            tables[row_id] = {
                "strategy": strategy,
                "entries": entries_b,
                "table": memory.subset(keys, budget=entries_b),
            }
    tables["full"] = {
        "strategy": "full",
        "entries": n_entries,
        "table": memory,
    }

    def key_set_hash(table: Memory) -> str:
        digest = hashlib.sha1()
        for key in sorted(table.entries):
            digest.update(",".join(str(int(t)) for t in key).encode("utf-8"))
            digest.update(b";")
        return digest.hexdigest()[:16]

    acc = {
        row_id: {"hits": 0, "accepted": 0, "fired": 0}
        for row_id in tables
    }
    per_case_delta: dict[str, list[float]] = {row_id: [] for row_id in tables}
    case_base_nll: list[float] = []
    case_fused_nll: dict[str, list[float]] = {row_id: [] for row_id in tables}
    case_groups: list[str] = []
    total_positions = 0

    for index, case in enumerate(test_task.cases):
        prompt = list(case.prompt_ids)
        reference = list(case.reference_ids)
        if not reference:
            continue
        logits, _ = model.forward(prompt + reference[:-1])
        ctx = list(prompt)
        base_case = 0.0
        fused_case = {row_id: 0.0 for row_id in tables}
        delta_case = {row_id: 0.0 for row_id in tables}
        for i, token in enumerate(reference):
            total_positions += 1
            p = softmax(logits[len(prompt) - 1 + i])
            base_nll = -float(np.log(max(float(p[token]), 1e-12)))
            base_case += base_nll
            argmax = int(p.argmax())
            top_prob = float(p.max())
            for row_id, meta in tables.items():
                fused_nll = base_nll
                hit = meta["table"].lookup(ctx)
                if hit is not None:
                    key, entry = hit
                    acc[row_id]["hits"] += 1
                    if entry.continuation and int(entry.continuation[0]) == argmax:
                        acc[row_id]["accepted"] += 1
                    if gate.allows(key, entry, top_prob):
                        acc[row_id]["fired"] += 1
                        q = memory_dist(entry, p.shape[0])
                        fused = fuse_probs(p, q, lam, "arithmetic")
                        fused_nll = -float(
                            np.log(max(float(fused[token]), 1e-12))
                        )
                fused_case[row_id] += fused_nll
                delta_case[row_id] += base_nll - fused_nll
            ctx.append(token)
        length = len(reference)
        case_groups.append(case.origin or f"case-{index}")
        case_base_nll.append(base_case / length)
        for row_id in tables:
            case_fused_nll[row_id].append(fused_case[row_id] / length)
            per_case_delta[row_id].append(delta_case[row_id] / length)

    baseline_nll = (
        float(np.mean(case_base_nll)) if case_base_nll else 0.0
    )
    rows: list[dict] = []
    for row_id, meta in tables.items():
        deltas = per_case_delta[row_id]
        stats = clustered_delta_stats(
            list(zip(case_groups, deltas)), bootstrap=bootstrap, seed=seed
        )
        positions = total_positions or 1
        hits = acc[row_id]["hits"]
        row = {
            "row_id": row_id,
            "strategy": meta["strategy"],
            "entries": meta["entries"],
            "by_order": meta["table"].stats()["by_order"],
            "key_hash": key_set_hash(meta["table"]),
            "hit_rate": round(hits / positions, 6),
            "accept_rate": (
                round(acc[row_id]["accepted"] / hits, 6) if hits else None
            ),
            "accept_coverage": round(acc[row_id]["accepted"] / positions, 6),
            "fired_rate": round(acc[row_id]["fired"] / positions, 6),
            "mean": round(stats["mean"], 6),
            "se": round(stats["se"], 6),
            "ci_low": None if stats["ci_low"] is None else round(stats["ci_low"], 6),
            "ci_high": None if stats["ci_high"] is None else round(stats["ci_high"], 6),
            "improved_groups": stats["improved_groups"],
            "n_test_files": stats["n_groups"],
            "n_test_cases": stats["n_cases"],
            "nll": (
                round(float(np.mean(case_fused_nll[row_id])), 6)
                if case_fused_nll[row_id]
                else 0.0
            ),
            "baseline_nll": round(baseline_nll, 6),
        }
        rows.append(row)

    pairwise: list[dict] = []
    for entries_b in budget_values:
        for i, a in enumerate(strategies):
            for b in strategies[i + 1 :]:
                row_a, row_b = f"{a}@{entries_b}", f"{b}@{entries_b}"
                if row_a not in per_case_delta or row_b not in per_case_delta:
                    continue
                diffs = [
                    x - y
                    for x, y in zip(
                        per_case_delta[row_a], per_case_delta[row_b]
                    )
                ]
                stats = clustered_delta_stats(list(zip(case_groups, diffs)))
                pairwise.append(
                    {
                        "entries": entries_b,
                        "a": a,
                        "b": b,
                        "mean_diff": round(stats["mean"], 6),
                        "se": round(stats["se"], 6),
                        "t": (
                            round(stats["mean"] / stats["se"], 3)
                            if stats["se"] > 0
                            else None
                        ),
                    }
                )

    overlap: list[dict] = []
    for entries_b in budget_values:
        quality_keys = set(
            select_keys(memory, usage, strategy="quality", budget=entries_b)
        )
        speed_keys = set(
            select_keys(memory, usage, strategy="speed", budget=entries_b)
        )
        union = quality_keys | speed_keys
        overlap.append(
            {
                "entries": entries_b,
                "overlap": (
                    round(len(quality_keys & speed_keys) / len(union), 6)
                    if union
                    else 0.0
                ),
                "intersection": len(quality_keys & speed_keys),
                "union": len(union),
            }
        )

    result["budgets"] = budget_values
    result["positions"] = total_positions
    result["rows"] = rows
    result["pairwise"] = pairwise
    result["overlap"] = overlap
    return result


def plot_dose_response(report: dict, path: str | Path) -> None:
    """剂量-反应:质量增益对记忆表条目数,误差棒是自助法区间。

    同规模的第二次抽样(memory_offset > 0)用空心标记单独画出,
    便于一眼看出"换一批文件建表"带来的抖动有多大。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    groups: dict[str, list[dict]] = {}
    for row in report.get("rows", []):
        if row.get("gated_mean") is None:
            continue
        groups.setdefault(str(row["corpus"]), []).append(row)
    for corpus, rows in groups.items():
        main = sorted(
            (row for row in rows if int(row.get("memory_offset", 0)) == 0),
            key=lambda row: row["entries"],
        )
        if main:
            xs = [row["entries"] for row in main]
            ys = [row["gated_mean"] for row in main]
            err = [
                [max(0.0, row["gated_mean"] - row["gated_ci_low"]) for row in main],
                [max(0.0, row["gated_ci_high"] - row["gated_mean"]) for row in main],
            ]
            ax.errorbar(
                xs, ys, yerr=err, marker="o", capsize=4, label=f"{corpus} (main draw)"
            )
        extra = [row for row in rows if int(row.get("memory_offset", 0)) != 0]
        if extra:
            ax.errorbar(
                [row["entries"] for row in extra],
                [row["gated_mean"] for row in extra],
                yerr=[
                    [max(0.0, row["gated_mean"] - row["gated_ci_low"]) for row in extra],
                    [max(0.0, row["gated_ci_high"] - row["gated_mean"]) for row in extra],
                ],
                marker="s",
                linestyle="none",
                markerfacecolor="none",
                capsize=4,
                label=f"{corpus} (second draw)",
            )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("memory table entries")
    ax.set_xscale("log")
    ax.set_ylabel("gated quality gain G (NLL, file-clustered, 95% CI)")
    ax.set_title("dose-response: quality gain vs memory budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
