import numpy as np
import pytest

from everyngram.tasks import (
    CompletionTask,
    EvalCase,
    build_completion_task,
    build_corpus_ids,
    collect_code_files,
    next_token_accuracy,
    perplexity,
    split_files,
)


class FakeTokenizer:
    """每个字符映射到 0 到 15 的 token id。"""

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 16 for ch in text]


class OneHotCachedModel:
    """下一个 token 恒为 (当前 token + 1) % 16。"""

    def forward(self, token_ids, cache=None):
        out = np.zeros((len(token_ids), 16), dtype=np.float64)
        for i, token in enumerate(token_ids):
            out[i, (int(token) + 1) % 16] = 1.0
        return out, list(cache or []) + [int(t) for t in token_ids]


def test_collect_code_files_filters_and_is_deterministic(tmp_path):
    (tmp_path / "small.py").write_text("x = 1\n" * 50, encoding="utf-8")
    (tmp_path / "good.py").write_text("y = 2\n" * 500, encoding="utf-8")
    (tmp_path / "big.py").write_text("z" * 300_000, encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("w" * 5000, encoding="utf-8")

    files = collect_code_files(
        [tmp_path], max_files=10, max_bytes=200_000, min_bytes=2_000
    )

    assert [p.name for p in files] == ["good.py"]
    again = collect_code_files(
        [tmp_path], max_files=10, max_bytes=200_000, min_bytes=2_000
    )
    assert files == again


def test_split_files_keeps_holdout():
    from pathlib import Path

    files = [Path(f"f{i}.py") for i in range(5)]
    corpus, holdout = split_files(files, holdout_ratio=0.2)

    assert len(corpus) == 4
    assert len(holdout) == 1


def test_build_corpus_ids_concatenates(tmp_path):
    (tmp_path / "a.py").write_text("ab", encoding="utf-8")
    (tmp_path / "b.py").write_text("ba", encoding="utf-8")

    ids = build_corpus_ids([tmp_path / "a.py", tmp_path / "b.py"], FakeTokenizer())

    assert ids == [1, 2, 2, 1]


def test_build_completion_task_shapes(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("a" * 400, encoding="utf-8")

    task = build_completion_task(
        [path], FakeTokenizer(), n_cases=2, context_len=16, horizon=4, seed=0
    )

    assert task.name == "code_completion"
    assert len(task.cases) == 2
    for case in task.cases:
        assert len(case.prompt_ids) == 16
        assert len(case.reference_ids) == 4


def test_completion_score_is_exact_match():
    task = CompletionTask(
        name="t", cases=(EvalCase(prompt_ids=(1, 2), reference_ids=(3, 4)),)
    )
    case = task.cases[0]

    assert task.score([3, 4], case) == 1.0
    assert task.score([3, 9], case) == 0.0
    assert task.score([3], case) == 0.0


def test_next_token_accuracy_counts_argmax_hits():
    cases = (EvalCase(prompt_ids=(1,), reference_ids=(2,)),)
    assert next_token_accuracy(OneHotCachedModel(), cases) == 1.0

    bad = (EvalCase(prompt_ids=(1,), reference_ids=(9,)),)
    assert next_token_accuracy(OneHotCachedModel(), bad) == 0.0


def test_perplexity_of_uniform_model_equals_vocab_size():
    class UniformModel:
        def forward(self, token_ids, cache=None):
            return np.zeros((len(token_ids), 16), dtype=np.float64), None

    assert abs(perplexity(UniformModel(), [1, 2, 3, 4]) - 16.0) < 1e-6


def test_perplexity_requires_two_tokens():
    class UniformModel:
        def forward(self, token_ids, cache=None):
            return np.zeros((len(token_ids), 16), dtype=np.float64), None

    with pytest.raises(ValueError):
        perplexity(UniformModel(), [1])


def test_accuracy_with_lambda_zero_equals_baseline():
    from everyngram.fuse import FusionSpec
    from everyngram.memory import Memory

    model = OneHotCachedModel()
    cases = (EvalCase(prompt_ids=(1,), reference_ids=(2,)),)
    memory = Memory.build([1, 2] * 10, fingerprint="fp", orders=(1,), budget=100)

    base = next_token_accuracy(model, cases)
    zero = next_token_accuracy(model, cases, fusion=FusionSpec(lam=0.0, memory=memory))

    assert base == zero == 1.0


def test_accuracy_follows_memory_when_lambda_is_one():
    from everyngram.fuse import FusionSpec
    from everyngram.memory import Memory

    model = OneHotCachedModel()          # 模型预测 (1+1)%16 = 2
    memory = Memory.build([1, 9] * 10, fingerprint="fp", orders=(1,), budget=100)
    spec = FusionSpec(lam=1.0, memory=memory)

    hits = (EvalCase(prompt_ids=(1,), reference_ids=(9,)),)
    misses = (EvalCase(prompt_ids=(1,), reference_ids=(2,)),)

    assert next_token_accuracy(model, hits, fusion=spec) == 1.0
    assert next_token_accuracy(model, misses, fusion=spec) == 0.0


def test_perplexity_with_perfect_memory_approaches_one():
    from everyngram.fuse import FusionSpec
    from everyngram.memory import Memory

    class UniformModel:
        def forward(self, token_ids, cache=None):
            return np.zeros((len(token_ids), 16), dtype=np.float64), None

    ids = [1, 2, 3, 4, 5]
    memory = Memory.build(ids * 4, fingerprint="fp", orders=(1,), budget=100)
    spec = FusionSpec(lam=1.0, memory=memory)

    assert perplexity(UniformModel(), ids, fusion=spec) < 1.05


def test_tv_distance_is_zero_when_memory_matches_model():
    from everyngram.memory import Memory
    from everyngram.tasks import tv_distance

    class PeakedModel:
        """词表 3:每个位置的分布是 one-hot 于 token 1。"""

        def forward(self, token_ids, cache=None):
            row = np.array([-np.inf, 0.0, -np.inf])
            return np.tile(row, (len(token_ids), 1)), None

    # 键 (0,) 只续写 1,与模型的 one-hot 完全一致
    memory = Memory.build([0, 1] * 10, fingerprint="fp", orders=(1,), budget=100)
    cases = (EvalCase(prompt_ids=(0,), reference_ids=(1,)),)

    out = tv_distance(PeakedModel(), cases, memory)

    assert out["hit_rate"] == 1.0
    assert abs(out["tv_mean"]) < 1e-12


def test_tv_distance_skips_missing_keys():
    from everyngram.memory import Memory
    from everyngram.tasks import tv_distance

    model = OneHotCachedModel()
    memory = Memory.build([7] * 20, fingerprint="fp", orders=(1,), budget=100)
    cases = (EvalCase(prompt_ids=(1,), reference_ids=(2,)),)

    out = tv_distance(model, cases, memory)

    assert out["n_hits"] == 0
    assert out["hit_rate"] == 0.0
    assert out["tv_mean"] is None


def test_tv_distance_matches_p2_complement_on_constructed_pair():
    class FixedModel:
        """词表 3:第一个位置给出 logits,使 softmax 为 (0.6, 0.3, 0.1)。"""

        def forward(self, token_ids, cache=None):
            row = np.log(np.array([0.6, 0.3, 0.1]))
            return np.tile(row, (len(token_ids), 1)), None

    from everyngram.memory import Memory
    from everyngram.tasks import tv_distance

    # 键 (5,) 的续写计数是 {1: 5, 2: 5},归一化成 q=(0, 0.5, 0.5)
    memory = Memory.build(
        [5, 1, 5, 2] * 6, fingerprint="fp", orders=(1,), budget=100
    )
    cases = (EvalCase(prompt_ids=(5,), reference_ids=(1,)),)

    out = tv_distance(FixedModel(), cases, memory)

    # TV = 0.5*(|0.6-0|+|0.3-0.5|+|0.1-0.5|) = 0.6,补集 0.4 = Σmin(p,q)
    assert abs(out["tv_mean"] - 0.6) < 1e-9
    assert abs((1.0 - out["tv_mean"]) - 0.4) < 1e-9


def test_nll_of_uniform_model_equals_log_vocab():
    from everyngram.tasks import next_token_nll

    class UniformModel:
        def forward(self, token_ids, cache=None):
            return np.zeros((len(token_ids), 16), dtype=np.float64), None

    cases = (EvalCase(prompt_ids=(1, 2), reference_ids=(3, 4, 5)),)

    assert abs(next_token_nll(UniformModel(), cases) - np.log(16)) < 1e-9


def test_nll_with_lambda_zero_equals_baseline():
    from everyngram.fuse import FusionSpec
    from everyngram.memory import Memory
    from everyngram.tasks import next_token_nll

    model = OneHotCachedModel()                      # 预测 (token+1)%16
    memory = Memory.build([1, 2, 3] * 10, fingerprint="fp", orders=(1,), budget=100)
    cases = (EvalCase(prompt_ids=(1,), reference_ids=(2, 3)),)

    base = next_token_nll(model, cases)
    zero = next_token_nll(model, cases, fusion=FusionSpec(lam=0.0, memory=memory))

    assert abs(base - zero) < 1e-12


def test_nll_is_sensitive_while_accuracy_is_not():
    """分布被拉向真值但 argmax 不变:命中率不动,NLL 变好。"""
    from everyngram.fuse import FusionSpec
    from everyngram.memory import Entry, Memory
    from everyngram.tasks import next_token_accuracy, next_token_nll

    class FixedModel:
        """词表 3,logits 使 softmax 为 (0.2, 0.7, 0.1),argmax 恒为 1。"""

        def forward(self, token_ids, cache=None):
            row = np.log(np.array([0.2, 0.7, 0.1]))
            return np.tile(row, (len(token_ids), 1)), None

    memory = Memory(
        fingerprint="fp",
        orders=(1,),
        budget=1,
        continuation_len=1,
        next_top_k=1,
        entries={(7,): Entry(count=4, next_top=((0, 4),), continuation=(0,))},
    )
    cases = (EvalCase(prompt_ids=(7,), reference_ids=(0,)),)
    spec = FusionSpec(lam=0.2, memory=memory)        # λ=0.2 不足以翻转 argmax

    assert next_token_accuracy(FixedModel(), cases) == 0.0
    assert next_token_accuracy(FixedModel(), cases, fusion=spec) == 0.0
    assert next_token_nll(FixedModel(), cases, fusion=spec) < next_token_nll(
        FixedModel(), cases
    )


def test_paired_nll_delta_is_zero_for_identity_fusion():
    from everyngram.fuse import FusionSpec
    from everyngram.memory import Memory
    from everyngram.tasks import paired_nll_delta

    model = OneHotCachedModel()
    memory = Memory.build([1, 2, 3] * 10, fingerprint="fp", orders=(1,), budget=100)
    cases = (
        EvalCase(prompt_ids=(1,), reference_ids=(2, 3)),
        EvalCase(prompt_ids=(2,), reference_ids=(3, 4)),
    )

    out = paired_nll_delta(model, cases, fusion=FusionSpec(lam=0.0, memory=memory))

    assert out["n"] == 2
    assert abs(out["mean"]) < 1e-12
    assert out["se"] == 0.0
    assert out["improved"] == 0


def test_paired_nll_delta_positive_when_memory_helps():
    from everyngram.fuse import FusionSpec
    from everyngram.memory import Memory
    from everyngram.tasks import paired_nll_delta

    class UniformModel:
        def forward(self, token_ids, cache=None):
            return np.zeros((len(token_ids), 16), dtype=np.float64), None

    ids = [1, 2, 3, 4, 5]
    memory = Memory.build(ids * 4, fingerprint="fp", orders=(1,), budget=100)
    cases = (
        EvalCase(prompt_ids=(1,), reference_ids=(2,)),
        EvalCase(prompt_ids=(2,), reference_ids=(3,)),
    )

    out = paired_nll_delta(
        UniformModel(), cases, fusion=FusionSpec(lam=1.0, memory=memory)
    )

    assert out["mean"] > 0
    assert out["improved"] == 2
    assert out["se"] >= 0.0


def test_completion_exact_match_counts_full_horizon_hits():
    from everyngram.tasks import completion_exact_match

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

    hit = (EvalCase(prompt_ids=(1,), reference_ids=(2, 3, 4)),)
    miss = (EvalCase(prompt_ids=(1,), reference_ids=(2, 9, 4)),)

    assert completion_exact_match(CountingCachedModel(), hit, max_new_tokens=3) == 1.0
    assert completion_exact_match(CountingCachedModel(), miss, max_new_tokens=3) == 0.0
    assert completion_exact_match(CountingCachedModel(), (), max_new_tokens=3) == 0.0


def test_completion_prefix_score_grades_partial_matches():
    from everyngram.tasks import completion_prefix_score

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

    full = (EvalCase(prompt_ids=(1,), reference_ids=(2, 3, 4, 5)),)
    half = (EvalCase(prompt_ids=(1,), reference_ids=(2, 3, 9, 9)),)
    none = (EvalCase(prompt_ids=(1,), reference_ids=(9, 3, 4, 5)),)
    model = CountingCachedModel()

    assert completion_prefix_score(model, full, max_new_tokens=4) == 1.0
    assert completion_prefix_score(model, half, max_new_tokens=4) == 0.5
    assert completion_prefix_score(model, none, max_new_tokens=4) == 0.0
    assert completion_prefix_score(model, (), max_new_tokens=4) == 0.0


def test_clustered_delta_stats_averages_within_group_first():
    from everyngram.tasks import clustered_delta_stats

    pairs = [("a.py", 0.10), ("a.py", 0.20), ("b.py", 0.40)]

    out = clustered_delta_stats(pairs)

    assert out["n_cases"] == 3
    assert out["n_groups"] == 2
    assert abs(out["mean"] - 0.275) < 1e-12       # (0.15 + 0.40) / 2
    assert abs(out["se"] - 0.125) < 1e-12         # std([0.15, 0.40], ddof=1)/sqrt(2)
    assert out["improved_groups"] == 2
    naive = (0.10 + 0.20 + 0.40) / 3
    assert abs(naive - out["mean"]) > 0.04        # 聚类均值不等于用例均值


def test_clustered_delta_stats_is_safe_on_single_group():
    from everyngram.tasks import clustered_delta_stats

    out = clustered_delta_stats([("a.py", 0.5), ("a.py", -0.1)])

    assert out["n_groups"] == 1
    assert out["se"] == 0.0
    assert abs(out["mean"] - 0.2) < 1e-12
    assert out["improved_groups"] == 1


def test_collect_code_files_can_exclude_substrings(tmp_path):
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "a.py").write_text("x = 1\n" * 500, encoding="utf-8")
    skip = tmp_path / "site-packages"
    skip.mkdir()
    (skip / "b.py").write_text("y = 2\n" * 500, encoding="utf-8")

    files = collect_code_files(
        [tmp_path], max_files=10, exclude_substrings=("site-packages",)
    )

    assert [path.name for path in files] == ["a.py"]


def test_build_completion_task_records_origin_file(tmp_path):
    from everyngram.tasks import build_completion_task

    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("x = 1\n" * 200, encoding="utf-8")

    task = build_completion_task(
        [tmp_path / "a.py", tmp_path / "b.py"], FakeTokenizer(), n_cases=4
    )

    assert task.cases
    origins = {case.origin for case in task.cases}
    assert origins <= {str(tmp_path / "a.py"), str(tmp_path / "b.py")}
    assert all(case.origin for case in task.cases)


def test_bootstrap_interval_covers_zero_for_symmetric_noise():
    from everyngram.tasks import clustered_delta_stats

    pairs = [("a.py", 0.010)] * 5 + [("b.py", -0.010)] * 5

    out = clustered_delta_stats(pairs, bootstrap=1000, seed=0)

    assert out["bootstrap"] == 1000
    assert out["ci_low"] <= 0.0 <= out["ci_high"]


def test_bootstrap_interval_excludes_zero_for_consistent_positive_effect():
    from everyngram.tasks import clustered_delta_stats

    pairs = [(f"f{i}.py", 0.01 + 0.001 * i) for i in range(20)]

    out = clustered_delta_stats(pairs, bootstrap=1000, seed=0)

    assert out["ci_low"] > 0.0
    assert out["ci_low"] < out["mean"] < out["ci_high"]


def test_bootstrap_rejects_negative_resamples_and_zero_means_no_interval():
    import pytest

    from everyngram.tasks import clustered_delta_stats

    with pytest.raises(ValueError):
        clustered_delta_stats([("a.py", 0.1)], bootstrap=-1)

    out = clustered_delta_stats([("a.py", 0.1), ("b.py", 0.2)], bootstrap=0)

    assert out["bootstrap"] == 0
    assert out["ci_low"] is None and out["ci_high"] is None


def _table_text():
    return (
        "intro line\n"
        "\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |\n"
        "| 5 | 6 |\n"
        "\n"
        "tail prose without table\n"
    )


def test_collect_table_prompts_picks_data_rows_not_separator(tmp_path):
    from everyngram.tasks import collect_table_prompts

    path = tmp_path / "doc.txt"
    text = _table_text()
    path.write_text(text, encoding="utf-8")
    expected = [
        text.index("| 1 | 2 |"),
        text.index("| 3 | 4 |"),
        text.index("| 5 | 6 |"),
    ]

    prompts = collect_table_prompts(
        [path], FakeTokenizer(), n=10, context_len=4, tail_chars=40
    )

    assert len(prompts) == 3
    assert sorted(item["char_offset"] for item in prompts) == sorted(expected)
    for item in prompts:
        assert 1 <= len(item["prompt_ids"]) <= 4
        assert item["origin"].endswith("doc.txt")


def test_collect_table_prompts_is_deterministic_and_capped(tmp_path):
    from everyngram.tasks import collect_table_prompts

    path = tmp_path / "doc.txt"
    path.write_text(_table_text(), encoding="utf-8")

    first = collect_table_prompts([path], FakeTokenizer(), n=2, context_len=4)
    second = collect_table_prompts([path], FakeTokenizer(), n=2, context_len=4)

    assert first == second
    assert len(first) == 2


def test_collect_table_prompts_skips_prose_only_files(tmp_path):
    from everyngram.tasks import collect_table_prompts

    path = tmp_path / "prose.txt"
    path.write_text("just prose\n" * 50, encoding="utf-8")

    prompts = collect_table_prompts([path], FakeTokenizer(), n=5, context_len=4)

    assert prompts == ()


def test_collect_table_prompts_reports_reference_tokens(tmp_path):
    from everyngram.tasks import collect_table_prompts

    path = tmp_path / "doc.txt"
    text = _table_text()
    path.write_text(text, encoding="utf-8")

    prompts = collect_table_prompts(
        [path], FakeTokenizer(), n=1, context_len=4, horizon=5
    )

    assert len(prompts) == 1
    item = prompts[0]
    tail = text[item["char_offset"] :]
    expected = tuple(FakeTokenizer().encode(tail, add_special_tokens=False))[:5]
    assert item["reference_ids"] == expected
    assert 1 <= len(item["reference_ids"]) <= 5


def test_collect_table_docs_filters_and_ranks_by_table_lines(tmp_path):
    from everyngram.tasks import collect_table_docs

    rich = tmp_path / "rich.md"
    rich.write_text(
        "intro\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n" + "x" * 3000,
        encoding="utf-8",
    )
    meta = tmp_path / "pkg-1.0.dist-info"
    meta.mkdir()
    (meta / "METADATA").write_text(
        "Name: pkg\n\n| k | v |\n|---|---|\n| 1 | 2 |\n" + "y" * 3000,
        encoding="utf-8",
    )
    (tmp_path / "prose.md").write_text("no table\n" * 500, encoding="utf-8")
    (tmp_path / "tiny.md").write_text("| a |\n|---|\n| 1 |\n", encoding="utf-8")

    docs = collect_table_docs([tmp_path], max_files=10)

    assert [path.name for path in docs] == ["rich.md", "METADATA"]


def test_collect_table_docs_is_deterministic_and_capped(tmp_path):
    from everyngram.tasks import collect_table_docs

    for index in range(3):
        path = tmp_path / f"doc{index}.md"
        path.write_text(
            "| a | b |\n|---|---|\n| 1 | 2 |\n" + "z" * 3000, encoding="utf-8"
        )

    first = collect_table_docs([tmp_path], max_files=2)
    second = collect_table_docs([tmp_path], max_files=2)

    assert first == second
    assert len(first) == 2


def test_collect_table_docs_supports_patterns_and_excludes(tmp_path):
    from everyngram.tasks import collect_table_docs

    body = "| a | b |\n|---|---|\n| 1 | 2 |\n" + "x" * 3000
    (tmp_path / "keep.py").write_text(body, encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_x.py").write_text(body, encoding="utf-8")
    (tmp_path / "doc.md").write_text(body, encoding="utf-8")

    docs = collect_table_docs(
        [tmp_path],
        max_files=10,
        patterns=("*.py",),
        exclude_substrings=("tests",),
    )

    assert [path.name for path in docs] == ["keep.py"]
