import time

from everyngram.bench import (
    BenchReport,
    conclude,
    loop_overhead_ms,
    lookup_latency,
    measure_latency,
    measure_peak_rss,
    predict_speedup,
    table_readings,
)
from everyngram.memory import Memory

CORPUS = [1, 2, 3, 1, 2, 3, 1, 2, 4]


def test_table_readings_reports_file_bytes(tmp_path):
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=100)
    path = tmp_path / "m.ngmem"
    mem.save(path)

    readings = table_readings(mem, path)

    assert readings["entries"] == mem.stats()["entries"]
    assert readings["file_bytes"] == path.stat().st_size
    assert readings["file_mb"] > 0


def test_table_readings_works_without_path():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1,), budget=100)
    readings = table_readings(mem)

    assert readings["entries"] == 3
    assert readings["file_bytes"] is None


def test_lookup_latency_percentiles_ordered():
    mem = Memory.build(CORPUS, fingerprint="fp", orders=(1, 2, 3), budget=100)
    contexts = [[1, 2], [9, 9], [1, 2, 3, 1]]

    readings = lookup_latency(mem, contexts, repeats=3)

    # 3 个上下文 × 3 次重复
    assert readings["n"] == 9
    assert 0 < readings["p50_us"] <= readings["p99_us"]


def test_measure_latency_warms_up_and_counts_calls():
    calls = {"n": 0}

    def work():
        calls["n"] += 1
        time.sleep(0.001)

    result = measure_latency(work, repeats=3, warmup=2)

    assert calls["n"] == 5
    assert result["repeats"] == 3
    assert result["median_s"] > 0
    assert result["min_s"] <= result["median_s"] <= result["max_s"]


def test_measure_peak_rss_returns_result_and_positive_peak():
    value, peak_mb = measure_peak_rss(sum, [1, 2, 3])

    assert value == 6
    assert peak_mb > 0


def test_loop_overhead_is_positive_and_small():
    o_ms = loop_overhead_ms(max_new_tokens=200)
    assert 0 < o_ms < 1.0


def test_speedup_prediction_matches_hand_calculation():
    values = {"c1_s": 0.010, "c_gamma_s": 0.030, "o_s": 0.001, "d_gamma_s": 0.0005}

    pred = predict_speedup(values, accept_length=3.0)

    # plain = c1 + o = 0.011 s/token
    # 合并循环(默认):spec = (c_gamma + o + d) / L = 0.0315 / 3 = 0.0105 s/token
    assert abs(pred["plain_ms_per_token"] - 11.0) < 1e-6
    assert abs(pred["spec_ms_per_token"] - 10.5) < 1e-3
    assert abs(pred["predicted_speedup"] - 1.047619) < 1e-3

    legacy = predict_speedup(values, accept_length=3.0, merged_bonus=False)
    # 旧循环(验证 + bonus 两次前向):(c_gamma + c1 + o + d) / L = 0.0415 / 3
    assert abs(legacy["spec_ms_per_token"] - 13.833333) < 1e-3
    assert abs(legacy["predicted_speedup"] - 0.795181) < 1e-3


def test_conclude_three_states():
    assert conclude([1.2, 0.9], consistency_ok=True) == "通过"
    assert conclude([0.9, 0.8], consistency_ok=True) == "负结果"
    assert conclude([1.2], consistency_ok=False) == "不成立"
    assert conclude([], consistency_ok=True) == "不成立"


def test_report_save_json_roundtrip(tmp_path):
    report = BenchReport(
        config={"model": "fake"},
        env={"device": "cpu"},
        readings={"R1": {"entries": 3}},
        account={"c1_ms": 0.1},
        tasks={"code": {"next_token_accuracy": 0.5}},
        conclusion="负结果",
    )
    path = tmp_path / "report.json"
    report.save_json(path)

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config"]["model"] == "fake"
    assert payload["conclusion"] == "负结果"


def test_report_plot_writes_png(tmp_path):
    readings = {
        "R6": {
            "baseline_ms_per_token": 1.0,
            "runs": {
                "2": {
                    "gamma": 2,
                    "speedup": 1.5,
                    "stats": {"accept_rate": 0.6, "accept_length": 2.5},
                },
                "4": {
                    "gamma": 4,
                    "speedup": 2.2,
                    "stats": {"accept_rate": 0.75, "accept_length": 3.5},
                },
            },
        }
    }
    report = BenchReport(
        config={}, env={}, readings=readings, account={}, tasks={}, conclusion="通过"
    )
    path = tmp_path / "curve.png"
    report.plot(path)

    assert path.exists() and path.stat().st_size > 0


def test_run_fusion_scan_structure_and_lambda_zero_baseline():
    import numpy as np

    from everyngram.bench import run_fusion_scan
    from everyngram.tasks import EvalCase

    class CountingCachedModel:
        def __init__(self) -> None:
            self.vocab = 16

        def _logits(self, token_ids):
            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens):], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    memory = Memory.build(
        [i % 16 for i in range(64)], fingerprint="fp", orders=(1,), budget=100
    )
    cases = (
        EvalCase(prompt_ids=(1, 2, 3), reference_ids=(4,)),
        EvalCase(prompt_ids=(5, 6, 7), reference_ids=(8,)),
    )

    scan = run_fusion_scan(
        CountingCachedModel(),
        memory,
        prompt_ids=[0],
        cases=cases,
        ood_ids=[1, 2, 3, 4],
        lambdas=(0.0, 0.5),
        modes=("arithmetic",),
        gamma=2,
        max_new_tokens=6,
        repeats=1,
    )

    assert "baseline" in scan and "rows" in scan
    assert len(scan["rows"]) == 2
    row0 = scan["rows"][0]
    assert row0["lam"] == 0.0
    assert row0["in_domain"] == scan["baseline"]["in_domain"]
    for key in ("accept_rate", "accept_length", "speedup", "ood_perplexity"):
        assert key in row0


def test_plot_tradeoff_writes_png(tmp_path):
    from everyngram.bench import plot_tradeoff

    scan = {
        "baseline": {"ms_per_token": 1.0, "in_domain": 0.5, "ood_perplexity": 40.0},
        "rows": [
            {"mode": "arithmetic", "lam": 0.0, "gamma": 4, "accept_rate": 0.03,
             "accept_length": 1.03, "ms_per_token": 1.1, "speedup": 0.9,
             "in_domain": 0.5, "ood_perplexity": 41.0},
            {"mode": "arithmetic", "lam": 0.4, "gamma": 4, "accept_rate": 0.25,
             "accept_length": 2.2, "ms_per_token": 0.7, "speedup": 1.4,
             "in_domain": 0.55, "ood_perplexity": 45.0},
            {"mode": "geometric", "lam": 0.4, "gamma": 4, "accept_rate": 0.3,
             "accept_length": 2.5, "ms_per_token": 0.6, "speedup": 1.6,
             "in_domain": 0.52, "ood_perplexity": 60.0},
        ],
    }
    path = tmp_path / "tradeoff.png"

    plot_tradeoff(scan, path)

    assert path.exists() and path.stat().st_size > 0


def test_run_budget_scan_reports_monotone_entries_and_lambda_zero_baseline():
    import numpy as np

    from everyngram.bench import run_budget_scan
    from everyngram.tasks import EvalCase

    class CountingCachedModel:
        def _logits(self, token_ids):
            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens):], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    token_ids = [i % 16 for i in range(64)]
    cases = (
        EvalCase(prompt_ids=(1, 2, 3), reference_ids=(4, 5)),
        EvalCase(prompt_ids=(5, 6, 7), reference_ids=(8, 9)),
    )

    scan = run_budget_scan(
        CountingCachedModel(),
        token_ids,
        fingerprint="fp",
        budgets=(2, 10),
        cases=cases,
        lambdas=(0.0, 0.2),
        gamma=2,
        max_new_tokens=4,
        repeats=1,
    )

    assert len(scan["rows"]) == 4
    entries = [row["entries"] for row in scan["rows"] if row["lam"] == 0.0]
    assert entries == sorted(entries)
    zero_row = next(row for row in scan["rows"] if row["lam"] == 0.0)
    assert abs(zero_row["nll"] - scan["baseline"]["nll"]) < 1e-12
    for key in (
        "tv_mean",
        "hit_rate",
        "accept_rate",
        "accept_length",
        "speedup",
        "nll_gain",
        "file_bytes",
    ):
        assert key in zero_row


def test_run_pool_experiment_scores_both_pools_on_same_cases():
    import numpy as np

    from everyngram.bench import run_pool_experiment
    from everyngram.select import select_pools
    from everyngram.tasks import EvalCase

    class CountingCachedModel:
        def _logits(self, token_ids):
            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens):], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    memory = Memory.build(
        [i % 16 for i in range(64)], fingerprint="fp", orders=(1,), budget=100
    )
    scored = [
        {
            "key": (i,),
            "true_next": (i + 1) % 16,
            "speed_score": float(i) / 16,
            "quality_score": float(16 - i) / 16,
        }
        for i in range(8)
    ]
    pools = select_pools(scored, k=3)
    cases = (
        EvalCase(prompt_ids=(1, 2, 3), reference_ids=(4, 5)),
        EvalCase(prompt_ids=(5, 6, 7), reference_ids=(8, 9)),
    )

    result = run_pool_experiment(
        CountingCachedModel(),
        memory,
        pools,
        cases=cases,
        gamma=2,
        lambdas=(0.0, 0.2),
        max_new_tokens=4,
        repeats=1,
    )

    assert set(result["pools"]) == {"speed_pool", "quality_pool", "union_pool"}
    assert 0.0 <= result["overlap"] <= 1.0
    for name, block in result["pools"].items():
        for key in ("entries", "tv_mean", "hit_rate"):
            assert key in block
        rows = block["rows"]
        assert len(rows) == 2
        for key in (
            "lam",
            "nll_gain",
            "accept_rate",
            "accept_length",
            "speedup",
        ):
            assert key in rows[0]
    union = result["pools"]["union_pool"]
    assert union["entries"] >= result["pools"]["speed_pool"]["entries"]


def test_plot_budget_scan_writes_png(tmp_path):
    from everyngram.bench import plot_budget_scan

    scan = {
        "baseline": {"ms_per_token": 40.0, "nll": 2.0},
        "rows": [
            {"budget": 1000, "lam": 0.0, "entries": 1000, "tv_mean": 0.5,
             "hit_rate": 0.4, "accept_rate": 0.03, "accept_length": 1.1,
             "speedup": 0.5, "nll": 2.0, "nll_gain": 0.0},
            {"budget": 1000, "lam": 0.2, "entries": 1000, "tv_mean": 0.5,
             "hit_rate": 0.4, "accept_rate": 0.1, "accept_length": 1.4,
             "speedup": 0.75, "nll": 1.9, "nll_gain": 0.1},
            {"budget": 5000, "lam": 0.2, "entries": 5000, "tv_mean": 0.6,
             "hit_rate": 0.6, "accept_rate": 0.12, "accept_length": 1.5,
             "speedup": 0.8, "nll": 1.85, "nll_gain": 0.15},
        ],
    }
    path = tmp_path / "budget.png"

    plot_budget_scan(scan, path)

    assert path.exists() and path.stat().st_size > 0


def test_plot_pool_experiment_writes_png(tmp_path):
    from everyngram.bench import plot_pool_experiment

    result = {
        "baseline": {"nll": 2.0, "ms_per_token": 40.0},
        "overlap": 0.1,
        "k_effective": 8,
        "overlap_curve": {"ks": [2, 4, 8], "overlap": [0.0, 0.0, 1.0],
                          "intersection": [0, 0, 8]},
        "pools": {
            "speed_pool": {"entries": 8, "tv_mean": 0.2, "hit_rate": 0.3,
                           "rows": [{"lam": 0.2, "accept_rate": 0.2,
                                     "accept_length": 1.8, "speedup": 0.9,
                                     "nll": 2.0, "nll_gain": 0.0}]},
            "quality_pool": {"entries": 8, "tv_mean": 0.9, "hit_rate": 0.3,
                             "rows": [{"lam": 0.2, "accept_rate": 0.01,
                                       "accept_length": 1.0, "speedup": 0.5,
                                       "nll": 1.5, "nll_gain": 0.5}]},
        },
    }
    path = tmp_path / "pool.png"

    plot_pool_experiment(result, path)

    assert path.exists() and path.stat().st_size > 0


def test_run_gated_experiment_freezes_dev_choice_and_scores_test():
    import numpy as np

    from everyngram.bench import run_gated_experiment
    from everyngram.fuse import Gate
    from everyngram.tasks import EvalCase

    class CountingCachedModel:
        def _logits(self, token_ids):
            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens):], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    memory = Memory.build(
        [i % 16 for i in range(64)], fingerprint="fp", orders=(1,), budget=100
    )
    dev = (EvalCase(prompt_ids=(1, 2), reference_ids=(3, 4)),)
    test = (
        EvalCase(prompt_ids=(1, 2), reference_ids=(3, 4)),
        EvalCase(prompt_ids=(5, 6), reference_ids=(7, 8)),
    )

    out = run_gated_experiment(
        CountingCachedModel(),
        memory,
        dev_cases=dev,
        test_cases=test,
        gates=(Gate(min_order=1),),
        lambdas=(0.05,),
        gamma=2,
        em_tokens=4,
        timing_tokens=4,
        repeats=1,
    )

    assert out["selection"]["best"] is not None
    for name in ("gated", "ungated_same_lam", "ungated_lam_0.2"):
        assert name in out["test"]["configs"]
    gated = out["test"]["configs"]["gated"]
    for key in ("nll_delta", "accuracy", "em", "speedup", "ms_per_token"):
        assert key in gated
    assert gated["nll_delta"]["n"] == 2
    assert "nll" in out["test"]["baseline"]


def test_plot_gated_experiment_writes_png(tmp_path):
    from everyngram.bench import plot_gated_experiment

    result = {
        "selection": {
            "best": {
                "gate": {"min_order": 2, "min_share": 0.5, "max_top_prob": 0.9},
                "lam": 0.05,
                "dev_gain": 0.03,
                "coverage": 0.4,
            }
        },
        "test": {
            "baseline": {"nll": 1.6, "accuracy": 0.5, "em": 0.2, "ms_per_token": 40.0},
            "configs": {
                "gated": {
                    "gate": {"min_order": 2, "min_share": 0.5, "max_top_prob": 0.9},
                    "lam": 0.05,
                    "nll_delta": {"n": 16, "mean": 0.03, "se": 0.01, "improved": 10},
                    "accuracy": 0.5, "em": 0.25, "ms_per_token": 42.0, "speedup": 0.95,
                    "accept_rate": 0.1, "accept_length": 1.2,
                },
                "ungated_same_lam": {
                    "gate": None, "lam": 0.05,
                    "nll_delta": {"n": 16, "mean": -0.01, "se": 0.01, "improved": 5},
                    "accuracy": 0.5, "em": 0.2, "ms_per_token": 43.0, "speedup": 0.93,
                    "accept_rate": 0.12, "accept_length": 1.25,
                },
            },
        },
    }
    path = tmp_path / "gated.png"

    plot_gated_experiment(result, path)

    assert path.exists() and path.stat().st_size > 0


def test_make_folds_partitions_files_without_overlap():
    from everyngram.bench import make_folds

    files = [f"f{i}.py" for i in range(12)]

    folds = make_folds(files, k=4, seed=0)

    assert len(folds) == 4
    tests = [set(fold["test"]) for fold in folds]
    assert set().union(*tests) == set(files)
    for i in range(len(tests)):
        for j in range(i + 1, len(tests)):
            assert not (tests[i] & tests[j])
    for fold in folds:
        assert set(fold["corpus"]).isdisjoint(fold["test"])
        assert set(fold["corpus"]).isdisjoint(fold["dev"])
        assert len(fold["test"]) == 3


def test_make_folds_requires_enough_files():
    import pytest

    from everyngram.bench import make_folds

    with pytest.raises(ValueError):
        make_folds(["a.py", "b.py"], k=2)


def test_make_folds_rejects_two_folds_because_corpus_would_be_empty():
    import pytest

    from everyngram.bench import make_folds

    with pytest.raises(ValueError, match="折数"):
        make_folds([f"f{i}.py" for i in range(9)], k=2)


def test_make_folds_with_three_folds_keeps_corpus_non_empty():
    from everyngram.bench import make_folds

    files = [f"f{i}.py" for i in range(9)]

    folds = make_folds(files, k=3, seed=0)

    for fold in folds:
        assert len(fold["test"]) == 3
        assert len(fold["dev"]) == 3
        assert len(fold["corpus"]) == 3


def test_plot_fold_experiment_writes_png(tmp_path):
    from everyngram.bench import plot_fold_experiment

    report = {
        "corpora": {
            "stdlib": {
                "pooled": {
                    "gated": {
                        "n_folds": 3,
                        "mean_across_folds": 0.004,
                        "se_across_folds": 0.002,
                        "per_fold_delta": [0.005, 0.002, 0.006],
                        "per_fold_improved": 3,
                    }
                }
            },
            "site": {
                "skipped": "文件数不足 12",
            },
        }
    }
    path = tmp_path / "folds.png"

    plot_fold_experiment(report, path)

    assert path.exists() and path.stat().st_size > 0


def test_run_power_experiment_freezes_selection_and_scores_many_files(tmp_path):
    import numpy as np

    from everyngram.bench import run_power_experiment
    from everyngram.fuse import Gate

    class CountingCachedModel:
        def _logits(self, token_ids):
            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens):], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 16 for ch in text]

    for i in range(30):
        # 每个 2400 字节,高于 collect_code_files 的 2000 字节下限
        (tmp_path / f"f{i:03d}.py").write_text("x = 1\n" * 400, encoding="utf-8")

    out = run_power_experiment(
        CountingCachedModel(),
        FakeTokenizer(),
        corpus_root=str(tmp_path),
        fingerprint="fp",
        gates=(Gate(min_order=1),),
        lambdas=(0.02,),
        memory_files=6,
        dev_files=6,
        test_files=18,
        dev_cases_per_file=2,
        test_cases_per_file=2,
        em_tokens=1,
    )

    assert out["selected"] is not None
    assert out["counts"]["memory"] == 6
    assert out["counts"]["dev"] == 6
    assert out["counts"]["test"] == 18
    gated = out["test"]["gated"]
    for key in ("clustered", "ci_low", "ci_high", "em1", "n_test_files"):
        assert key in gated
    assert gated["n_test_files"] > 0
    # 逐文件差值要落进结果,供跨 run 的配对比较(增量 8 的判据需要)
    assert "per_file_deltas" in gated
    assert len(gated["per_file_deltas"]) == gated["n_test_files"]


def test_run_power_experiment_requires_enough_files(tmp_path):
    import pytest

    from everyngram.bench import run_power_experiment

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

    (tmp_path / "only.py").write_text("x = 1\n" * 500, encoding="utf-8")

    with pytest.raises(ValueError):
        run_power_experiment(
            object(),
            FakeTokenizer(),
            corpus_root=str(tmp_path),
            fingerprint="fp",
            gates=(),
            lambdas=(0.02,),
            memory_files=6,
            dev_files=6,
            test_files=18,
        )


def test_run_fold_experiment_reports_per_fold_and_pooled_stats(tmp_path):
    import numpy as np

    from everyngram.bench import make_folds, run_fold_experiment
    from everyngram.fuse import Gate

    class CountingCachedModel:
        def _logits(self, token_ids):
            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens):], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 16 for ch in text]

    files = []
    for i in range(12):
        path = tmp_path / f"f{i}.py"
        path.write_text("x = 1\n" * 200, encoding="utf-8")
        files.append(str(path))
    folds = make_folds(files, k=4, seed=0)

    out = run_fold_experiment(
        CountingCachedModel(),
        FakeTokenizer(),
        folds=folds,
        fingerprint="fp",
        gates=(Gate(min_order=1),),
        lambdas=(0.02,),
        dev_cases=4,
        test_cases=4,
        gamma=2,
        prefix_tokens=4,
        timing_tokens=4,
    )

    assert len(out["folds"]) == 4
    for fold in out["folds"]:
        assert fold["n_test_cases"] > 0
        assert "gated" in fold["test"]["configs"]
    pooled = out["pooled"]["gated"]
    for key in (
        "n_folds",
        "mean_across_folds",
        "se_across_folds",
        "per_fold_delta",
        "per_fold_improved",
    ):
        assert key in pooled


def test_run_power_experiment_nests_memory_and_keeps_test_split_fixed(tmp_path):
    """同一 pool/seed 下,不同记忆表规模必须是嵌套子集,dev/test 必须完全一致。"""
    import numpy as np

    from everyngram.bench import run_power_experiment
    from everyngram.fuse import Gate

    class CountingCachedModel:
        def _logits(self, token_ids):
            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens):], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 16 for ch in text]

    for i in range(40):
        (tmp_path / f"f{i:03d}.py").write_text("x = 1\n" * 400, encoding="utf-8")

    common = dict(
        corpus_root=str(tmp_path),
        fingerprint="fp",
        gates=(Gate(min_order=1),),
        lambdas=(0.02,),
        pool_files=20,
        dev_files=6,
        test_files=10,
        dev_cases_per_file=2,
        test_cases_per_file=2,
    )
    small = run_power_experiment(
        CountingCachedModel(), FakeTokenizer(), memory_files=4, **common
    )
    large = run_power_experiment(
        CountingCachedModel(), FakeTokenizer(), memory_files=8, **common
    )

    assert small["split"]["test"] == large["split"]["test"]
    assert small["split"]["dev"] == large["split"]["dev"]
    assert set(small["split"]["memory"]) < set(large["split"]["memory"])
    assert small["counts"]["memory"] == 4
    assert large["counts"]["memory"] == 8


def test_plot_dose_response_writes_png(tmp_path):
    from everyngram.bench import plot_dose_response

    report = {
        "rows": [
            {"name": "site-20", "corpus": "site-packages", "memory_files": 20,
             "memory_offset": 0, "entries": 50000, "gated_mean": 0.01,
             "gated_ci_low": 0.002, "gated_ci_high": 0.018},
            {"name": "site-40", "corpus": "site-packages", "memory_files": 40,
             "memory_offset": 0, "entries": 100000, "gated_mean": 0.03,
             "gated_ci_low": 0.02, "gated_ci_high": 0.04},
            {"name": "std-40-draw2", "corpus": "stdlib", "memory_files": 40,
             "memory_offset": 20, "entries": 90000, "gated_mean": 0.0,
             "gated_ci_low": -0.01, "gated_ci_high": 0.01},
            {"name": "std-40", "corpus": "stdlib", "memory_files": 40,
             "memory_offset": 0, "entries": 92000, "gated_mean": 0.02,
             "gated_ci_low": 0.01, "gated_ci_high": 0.03},
        ]
    }
    path = tmp_path / "dose.png"

    plot_dose_response(report, path)

    assert path.exists() and path.stat().st_size > 0


def _budget_toy_files(tmp_path, count=24):
    import random

    for i in range(count):
        rng = random.Random(1000 + i)
        text = "".join(rng.choice("abcdefghijklmnop ") for _ in range(2400))
        (tmp_path / f"f{i:03d}.py").write_text(text, encoding="utf-8")


class _BudgetCountingModel:
    """下一个 token 恒为 (当前 token + 1) % 16。"""

    def _logits(self, token_ids):
        import numpy as np

        out = np.zeros((len(token_ids), 16), dtype=np.float64)
        for i, token in enumerate(token_ids):
            out[i, (int(token) + 1) % 16] = 1.0
        return out

    def forward(self, token_ids, cache=None):
        tokens = [] if cache is None else list(cache)
        new = [int(t) for t in token_ids]
        return self._logits(tokens + new)[len(tokens) :], tokens + new

    def crop(self, cache, keep: int):
        return list(cache)[:keep]


class _BudgetTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 16 for ch in text]


def _run_budget_toy(tmp_path, **overrides):
    from everyngram.bench import run_budget_selection_experiment
    from everyngram.fuse import Gate

    _budget_toy_files(tmp_path)
    kwargs = dict(
        corpus_root=str(tmp_path),
        fingerprint="fp",
        gates=(Gate(min_order=1),),
        lambdas=(0.02,),
        pool_files=6,
        dev_files=6,
        test_files=12,
        dev_cases_per_file=2,
        test_cases_per_file=2,
        budgets=(8,),
        bootstrap=100,
    )
    kwargs.update(overrides)
    return run_budget_selection_experiment(
        _BudgetCountingModel(), _BudgetTokenizer(), **kwargs
    )


def test_run_budget_selection_full_row_matches_paired_nll_delta(tmp_path):
    from pathlib import Path

    from everyngram.fuse import FusionSpec, Gate
    from everyngram.memory import Memory
    from everyngram.tasks import (
        build_completion_task,
        build_corpus_ids,
        paired_nll_delta,
    )

    out = _run_budget_toy(tmp_path)

    assert out["selected"] is not None
    full = next(row for row in out["rows"] if row["row_id"] == "full")
    model, tok = _BudgetCountingModel(), _BudgetTokenizer()
    test_paths = [Path(p) for p in out["split_paths"]["test"]]
    corpus = build_corpus_ids(
        [Path(p) for p in out["split_paths"]["memory"]], tok
    )
    memory = Memory.build(corpus, fingerprint="fp", orders=(1, 2, 3), budget=1_000_000)
    task = build_completion_task(test_paths, tok, n_cases=len(test_paths) * 2)
    spec = FusionSpec(
        lam=float(out["selected"]["lam"]),
        mode="arithmetic",
        memory=memory,
        gate=Gate(**out["selected"]["gate"]),
    )
    reference = paired_nll_delta(model, task.cases, fusion=spec)

    # 报告里 mean 保留 6 位小数,等价性按半个单位(5e-7)的容差判
    assert abs(full["mean"] - reference["mean"]) < 1e-6
    assert full["entries"] == memory.stats()["entries"]
    # 不变量:全融 NLL = 基线 NLL − 增益(未触发位置按基线计,不计入增益)
    assert abs(full["nll"] - (full["baseline_nll"] - full["mean"])) < 5e-6


def test_run_budget_selection_blocked_gate_yields_zero_gain(tmp_path):
    from everyngram.fuse import Gate

    out = _run_budget_toy(
        tmp_path,
        gates=(Gate(min_order=1, max_top_prob=0.05),),
        min_coverage=0.0,
    )

    full = next(row for row in out["rows"] if row["row_id"] == "full")
    assert full["fired_rate"] == 0.0
    assert abs(full["mean"]) < 1e-9
    assert abs(full["nll"] - full["baseline_nll"]) < 5e-6


def test_run_budget_selection_reports_strategies_and_pairwise(tmp_path):
    out = _run_budget_toy(tmp_path)

    rows = out["rows"]
    strategies = {row["strategy"] for row in rows}
    assert {"freq", "query", "quality", "speed", "hybrid", "full"} <= strategies
    budget_rows = [row for row in rows if row["strategy"] != "full"]
    assert len(budget_rows) == 5
    for row in budget_rows:
        assert row["entries"] == 8
        assert 0.0 <= row["hit_rate"] <= 1.0
        assert row["accept_coverage"] <= row["hit_rate"] + 1e-12
    assert len({row["key_hash"] for row in budget_rows}) >= 2
    assert len(out["pairwise"]) == 10
    assert len(out["overlap"]) == 1
    assert 0.0 <= out["overlap"][0]["overlap"] <= 1.0
    assert out["dev_usage"]["keys_total"] == out["counts"]["entries"]
    assert out["dev_usage"]["positions"] > 0


def test_run_budget_selection_requires_enough_files(tmp_path):
    import pytest

    from everyngram.bench import run_budget_selection_experiment
    from everyngram.fuse import Gate

    (tmp_path / "only.py").write_text("x = 1\n" * 500, encoding="utf-8")

    with pytest.raises(ValueError):
        run_budget_selection_experiment(
            _BudgetCountingModel(),
            _BudgetTokenizer(),
            corpus_root=str(tmp_path),
            fingerprint="fp",
            gates=(Gate(min_order=1),),
            lambdas=(0.02,),
            pool_files=6,
            dev_files=6,
            test_files=12,
        )


def test_plot_budget_selection_writes_png(tmp_path):
    from everyngram.bench import plot_budget_selection

    rows = []
    for entries, scale in ((500, 1.0), (2000, 1.4)):
        for strategy, mean, coverage, hit in (
            ("freq", 0.0005, 0.001, 0.01),
            ("query", 0.012, 0.02, 0.21),
            ("quality", 0.018, 0.03, 0.25),
            ("speed", 0.006, 0.09, 0.18),
            ("hybrid", 0.015, 0.07, 0.24),
        ):
            rows.append(
                {
                    "row_id": f"{strategy}@{entries}",
                    "strategy": strategy,
                    "entries": entries,
                    "mean": mean * scale,
                    "accept_coverage": coverage,
                    "hit_rate": hit,
                }
            )
    rows.append(
        {
            "row_id": "full",
            "strategy": "full",
            "entries": 120000,
            "mean": 0.028,
            "accept_coverage": 0.09,
            "hit_rate": 0.62,
        }
    )
    path = tmp_path / "budget.png"

    plot_budget_selection({"rows": rows}, path)

    assert path.exists() and path.stat().st_size > 0


def test_run_copy_speed_experiment_reports_rows_and_summary():
    from everyngram.bench import run_copy_speed_experiment
    from everyngram.memory import Memory

    class CountingCachedModel:
        def _logits(self, token_ids):
            import numpy as np

            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens) :], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    memory = Memory.build(
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        fingerprint="fp",
        orders=(1,),
        budget=100,
    )
    scenarios = (
        {"name": "copy-0", "arm": "copy", "prompt_ids": (1, 2), "origin": "doc"},
        {"name": "free-0", "arm": "free", "prompt_ids": (4, 5), "origin": "instr"},
    )

    out = run_copy_speed_experiment(
        CountingCachedModel(),
        memory,
        scenarios,
        gammas=(2, 4),
        max_new_tokens=4,
        repeats=1,
        bootstrap=50,
    )

    assert len(out["rows"]) == 4
    for row in out["rows"]:
        assert row["identical"] is True
        assert row["speedup"] > 0
        assert row["accept_length"] >= 1.0
        assert 0.0 <= row["hit_rate"] <= 1.0
    assert set(out["summary"]) == {"copy", "free"}
    for arm in ("copy", "free"):
        assert set(out["summary"][arm]) == {"2", "4"}
        block = out["summary"][arm]["2"]
        assert block["n_scenarios"] == 1
        assert block["all_identical"] is True
        assert 0.0 <= block["frac_above_one"] <= 1.0
        assert block["mean_accept_length"] >= 1.0


def test_plot_copy_speed_writes_png(tmp_path):
    from everyngram.bench import plot_copy_speed

    rows = []
    for i in range(4):
        rows.append(
            {
                "name": f"copy-{i}",
                "arm": "copy",
                "gamma": 4,
                "accept_length": 2.2 + 0.3 * i,
                "speedup": 1.2 + 0.15 * i,
            }
        )
        rows.append(
            {
                "name": f"free-{i}",
                "arm": "free",
                "gamma": 4,
                "accept_length": 1.1 + 0.05 * i,
                "speedup": 0.6 + 0.02 * i,
            }
        )
    report = {
        "rows": rows,
        "summary": {
            "copy": {
                "4": {"mean": 1.4, "ci_low": 1.1, "ci_high": 1.7, "n_scenarios": 4}
            },
            "free": {
                "4": {"mean": 0.65, "ci_low": 0.6, "ci_high": 0.7, "n_scenarios": 4}
            },
        },
    }
    path = tmp_path / "copy_speed.png"

    plot_copy_speed(report, path)

    assert path.exists() and path.stat().st_size > 0


def test_run_adaptive_speed_experiment_reports_policies_and_summary():
    from everyngram.bench import run_adaptive_speed_experiment
    from everyngram.memory import Memory

    class CountingCachedModel:
        def _logits(self, token_ids):
            import numpy as np

            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens) :], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    memory = Memory.build(
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        fingerprint="fp",
        orders=(1,),
        budget=100,
    )
    scenarios = (
        {"name": "copy-0", "arm": "copy", "prompt_ids": (1,)},
        {"name": "free-0", "arm": "free", "prompt_ids": (5,)},
    )

    out = run_adaptive_speed_experiment(
        CountingCachedModel(),
        memory,
        scenarios,
        gamma=4,
        max_new_tokens=4,
        repeats=1,
        probe_steps=(2,),
        bootstrap=50,
    )

    assert len(out["rows"]) == 6
    policies = {row["policy"] for row in out["rows"]}
    assert policies == {"plain", "spec", "adaptive-k2"}
    for row in out["rows"]:
        assert row["identical"] is True
        assert row["ms"] > 0
        assert row["accept_length"] >= 1.0
    assert set(out["summary"]["totals"]) == policies
    assert set(out["summary"]["speedup_vs_plain"]) == policies
    assert set(out["summary"]["speedup_vs_plain_by_arm"]["copy"]) == policies
    assert "adaptive-k2" in out["summary"]["vs_spec_paired"]
    assert "adaptive-k2" in out["summary"]["auc"]
    assert 0.0 <= out["summary"]["switched_fraction"]["adaptive-k2"] <= 1.0
    assert out["summary"]["all_identical"] is True


def test_plot_adaptive_speed_writes_png(tmp_path):
    from everyngram.bench import plot_adaptive_speed

    rows = []
    for i in range(3):
        rows.append(
            {
                "name": f"copy-{i}",
                "arm": "copy",
                "policy": "plain",
                "probe_steps": None,
                "probe_rate": None,
                "switched": None,
                "accept_length": 1.0,
                "ms": 900.0,
            }
        )
        rows.append(
            {
                "name": f"copy-{i}",
                "arm": "copy",
                "policy": "spec",
                "probe_steps": None,
                "probe_rate": None,
                "switched": None,
                "accept_length": 1.2 + i,
                "ms": 700.0,
            }
        )
        rows.append(
            {
                "name": f"copy-{i}",
                "arm": "copy",
                "policy": "adaptive-k8",
                "probe_steps": 8,
                "probe_rate": 0.25 + 0.25 * i,
                "switched": i >= 1,
                "accept_length": 1.1 + i,
                "ms": 650.0,
            }
        )
    report = {
        "rows": rows,
        "summary": {
            "totals": {"plain": 2700.0, "spec": 2100.0, "adaptive-k8": 1950.0},
            "speedup_vs_plain": {"plain": 1.0, "spec": 1.286, "adaptive-k8": 1.385},
        },
    }
    path = tmp_path / "adaptive.png"

    plot_adaptive_speed(report, path)

    assert path.exists() and path.stat().st_size > 0


def test_ledger_measures_cached_and_nocache_costs_separately():
    from everyngram.bench import ledger, predict_speedup
    from everyngram.memory import Memory

    class RecordingCachedModel:
        def __init__(self):
            self.calls = []

        def forward(self, token_ids, cache=None):
            import numpy as np

            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            self.calls.append((len(new), len(tokens)))
            out = np.zeros((len(tokens) + len(new), 8), dtype=np.float64)
            for i, token in enumerate(tokens + new):
                out[i, (int(token) + 1) % 8] = 1.0
            return out[len(tokens) :], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    memory = Memory.build([1, 2, 3, 1, 2, 3], fingerprint="fp", orders=(1,), budget=100)
    contexts = [list(range(8))]
    model = RecordingCachedModel()

    values = ledger(model, memory, contexts, gamma=4, repeats=1)

    # 旧口径:无缓存、空 cache 上喂 1 / gamma 个 token
    assert (1, 0) in model.calls
    assert (4, 0) in model.calls
    # 新口径:真实上下文(长度 8)缓存上喂 1 / gamma 个 token
    assert (1, 8) in model.calls
    assert (4, 8) in model.calls
    for key in (
        "c1_s",
        "c_gamma_s",
        "c1_nocache_s",
        "c_gamma_nocache_s",
        "context_len",
    ):
        assert key in values
    assert values["context_len"] == 8
    prediction = predict_speedup(values, accept_length=3.0)
    assert prediction["spec_ms_per_token"] > 0


def test_run_task_layer_experiment_selects_lambda_and_reports_paired_stats():
    from everyngram.bench import run_task_layer_experiment
    from everyngram.memory import Memory

    class CountingCachedModel:
        def _logits(self, token_ids):
            import numpy as np

            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens) :], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    memory = Memory.build(
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        fingerprint="fp",
        orders=(1,),
        budget=100,
    )

    def scenario(i):
        return {
            "name": f"s{i}",
            "prompt_ids": (1,),
            "reference_ids": (2, 3, 4, 5),
        }

    dev = [scenario(i) for i in range(2)]
    test = [scenario(i) for i in range(2)]

    out = run_task_layer_experiment(
        CountingCachedModel(),
        memory,
        dev,
        test,
        lambdas=(0.2, 0.4),
        horizon=4,
        bootstrap=50,
    )

    assert set(out) == {"rows", "selected", "test_paired", "ood", "dev_spearman"}
    assert out["selected"]["lam"] in (0.2, 0.4)
    assert {row["split"] for row in out["rows"]} == {"dev", "test"}
    for row in out["rows"]:
        assert 0.0 <= row["mean_prefix"] <= 1.0
        assert 0.0 <= row["mean_changed"] <= 1.0
        assert len(row["per_scenario"]) == 2
    assert "mean" in out["test_paired"] and "se" in out["test_paired"]
    assert out["ood"] == []


def test_plot_task_layer_writes_png(tmp_path):
    from everyngram.bench import plot_task_layer

    rows = [
        {"split": "dev", "config": "plain", "lam": None, "mean_prefix": 0.28, "se_prefix": 0.05},
        {"split": "dev", "config": "fused-l0.2", "lam": 0.2, "mean_prefix": 0.32, "se_prefix": 0.05},
        {"split": "dev", "config": "fused-l0.4", "lam": 0.4, "mean_prefix": 0.40, "se_prefix": 0.06},
        {"split": "test", "config": "plain", "lam": None, "mean_prefix": 0.27, "se_prefix": 0.06},
        {"split": "test", "config": "fused-l0.2", "lam": 0.2, "mean_prefix": 0.31, "se_prefix": 0.06},
        {"split": "test", "config": "fused-l0.4", "lam": 0.4, "mean_prefix": 0.39, "se_prefix": 0.07},
    ]
    report = {
        "rows": rows,
        "ood": [
            {"lam": 0.2, "perplexity": 42.0, "ratio_vs_plain": 1.08},
            {"lam": 0.4, "perplexity": 48.0, "ratio_vs_plain": 1.23},
        ],
        "selected": {"lam": 0.4, "dev_prefix": 0.40, "dev_plain_prefix": 0.28},
    }
    path = tmp_path / "task_layer.png"

    plot_task_layer(report, path)

    assert path.exists() and path.stat().st_size > 0


def test_run_alignment_probe_measures_model_and_target_alignment():
    from everyngram.bench import run_alignment_probe
    from everyngram.memory import Memory

    class CountingCachedModel:
        def _logits(self, token_ids):
            import numpy as np

            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens) :], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    agreeing = Memory.build(
        [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
        fingerprint="fp",
        orders=(1,),
        budget=100,
    )
    disagreeing = Memory.build(
        [5, 9, 5, 9, 5, 9, 5, 9], fingerprint="fp", orders=(1,), budget=100
    )
    scenarios = [
        {"name": "s0", "prompt_ids": (1,), "reference_ids": (2, 3, 4, 5)},
        {"name": "s1", "prompt_ids": (1,), "reference_ids": (2, 3, 4, 5)},
    ]

    good = run_alignment_probe(CountingCachedModel(), agreeing, scenarios, horizon=4)

    assert len(good["rows"]) == 2
    assert good["summary"]["mean_a_model"] == 1.0
    assert good["summary"]["mean_a_target"] == 1.0
    assert good["summary"]["mean_hit_rate"] == 1.0

    bad = run_alignment_probe(
        CountingCachedModel(),
        disagreeing,
        [{"name": "s", "prompt_ids": (5,), "reference_ids": (6, 7, 8, 9)}],
        horizon=4,
    )

    assert bad["summary"]["mean_a_model"] == 0.0
    assert bad["summary"]["mean_a_target"] == 0.0


def test_run_alignment_probe_per_step_records_position_flags():
    from everyngram.bench import run_alignment_probe
    from everyngram.memory import Memory

    class CountingCachedModel:
        def _logits(self, token_ids):
            import numpy as np

            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens) :], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    agreeing = Memory.build(
        [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
        fingerprint="fp",
        orders=(1,),
        budget=100,
    )
    scenarios = [
        {"name": "s0", "prompt_ids": (1,), "reference_ids": (2, 3, 4, 5)},
    ]

    plain = run_alignment_probe(CountingCachedModel(), agreeing, scenarios, horizon=4)
    assert "step_agreements" not in plain["rows"][0]

    per_step = run_alignment_probe(
        CountingCachedModel(), agreeing, scenarios, horizon=4, per_step=True
    )
    row = per_step["rows"][0]
    assert row["step_agreements"] == [1, 1, 1, 1]
    assert per_step["summary"]["mean_a_model"] == plain["summary"]["mean_a_model"]


def test_task_layer_clusters_by_origin_when_present():
    from everyngram.bench import run_task_layer_experiment
    from everyngram.memory import Memory

    class CountingCachedModel:
        def _logits(self, token_ids):
            import numpy as np

            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 1) % 16] = 1.0
            return out

        def forward(self, token_ids, cache=None):
            tokens = [] if cache is None else list(cache)
            new = [int(t) for t in token_ids]
            return self._logits(tokens + new)[len(tokens) :], tokens + new

        def crop(self, cache, keep: int):
            return list(cache)[:keep]

    memory = Memory.build(
        [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
        fingerprint="fp",
        orders=(1,),
        budget=100,
    )
    dev = [
        {
            "name": "dev-0",
            "origin": "a.txt",
            "prompt_ids": (1,),
            "reference_ids": (2, 3),
        }
    ]
    test = [
        {
            "name": "t0",
            "origin": "a.txt",
            "prompt_ids": (1,),
            "reference_ids": (2, 3),
        },
        {
            "name": "t1",
            "origin": "b.txt",
            "prompt_ids": (1,),
            "reference_ids": (2, 3),
        },
        {
            "name": "t2",
            "origin": "b.txt",
            "prompt_ids": (1,),
            "reference_ids": (2, 3),
        },
    ]

    result = run_task_layer_experiment(
        CountingCachedModel(),
        memory,
        dev,
        test,
        lambdas=(0.5,),
        horizon=2,
        ood_ids=(),
        bootstrap=0,
    )

    assert result["test_paired"]["n_groups"] == 2
