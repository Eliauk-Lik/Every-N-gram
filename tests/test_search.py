import pytest


def _row(name, budget, *, task=None, speed=None, screened_out=None):
    row = {"name": name, "budget": budget, "task": task, "speed": speed}
    if screened_out:
        row["screened_out"] = screened_out
    return row


def _task(dev_prefix, lam, ood_ratio=1.05):
    return {
        "lam": lam,
        "dev_prefix": dev_prefix,
        "plain_prefix": 0.2,
        "ood_ratio": ood_ratio,
    }


def _speed(accept_length, accept_rate=0.2):
    return {"accept_length": accept_length, "accept_rate": accept_rate}


def test_select_best_task_respects_ood_constraint():
    from everyngram.search import select_best

    rows = [
        _row("a", 1000, task=_task(0.30, 0.5, ood_ratio=1.40)),
        _row("b", 1000, task=_task(0.10, 0.2, ood_ratio=1.05)),
    ]

    result = select_best(rows, "task", ood_max_ratio=1.10)

    assert result["name"] == "b"
    assert result["budget"] == 1000
    assert any(item["name"] == "a" and "OOD" in item["reason"] for item in result["rejected"])


def test_select_best_task_tie_breaks_by_smaller_lambda_then_budget():
    from everyngram.search import select_best

    rows = [
        _row("a", 2000, task=_task(0.30, 0.4)),
        _row("b", 1000, task=_task(0.30, 0.2)),
        _row("c", 1000, task=_task(0.30, 0.2)),
    ]
    rows[2]["name"] = "c"
    rows[2]["budget"] = 500

    result = select_best(rows, "task", ood_max_ratio=1.10)

    # 先 dev_prefix,再更小 λ,再更小预算: c(λ=0.2, 500) 胜 b(λ=0.2, 1000) 与 a
    assert result["name"] == "c"
    assert result["budget"] == 500


def test_select_best_speed_respects_accept_length_min():
    from everyngram.search import select_best

    rows = [
        _row("a", 1000, speed=_speed(3.5)),
        _row("b", 1000, speed=_speed(2.0)),
    ]

    result = select_best(rows, "speed", accept_length_min=2.5)

    assert result["name"] == "a"
    assert any(item["name"] == "b" and "接受长度" in item["reason"] for item in result["rejected"])


def test_select_best_speed_tie_breaks_by_budget():
    from everyngram.search import select_best

    rows = [
        _row("a", 8000, speed=_speed(2.5)),
        _row("b", 2000, speed=_speed(2.5)),
    ]

    result = select_best(rows, "speed")

    assert result["name"] == "b"
    assert result["budget"] == 2000


def test_select_best_both_requires_both_axes_and_ranks_task_first():
    from everyngram.search import select_best

    rows = [
        _row("a", 1000, task=_task(0.20, 0.3), speed=_speed(1.5)),
        _row("b", 1000, task=_task(0.20, 0.3), speed=_speed(2.5)),
        _row("c", 1000, task=_task(0.10, 0.2), speed=_speed(9.9)),
        _row("d", 1000, task=None, speed=_speed(9.9)),
    ]

    result = select_best(
        rows, "both", ood_max_ratio=1.10, accept_length_min=2.0
    )

    # c 的任务增益更低;d 没有 task;a 的接受长度低于下限 → b
    assert result["name"] == "b"
    rejected_names = {item["name"]: item["reason"] for item in result["rejected"]}
    assert "缺任务读数" in rejected_names["d"]
    assert "接受长度" in rejected_names["a"]


def test_select_best_skips_screened_out_rows():
    from everyngram.search import select_best

    rows = [
        _row("a", 1000, task=_task(0.9, 0.2), screened_out="a_target 低于筛选线"),
        _row("b", 1000, task=_task(0.1, 0.2)),
    ]

    result = select_best(rows, "task", ood_max_ratio=1.10)

    assert result["name"] == "b"
    assert any(
        item["name"] == "a" and "筛选" in item["reason"] for item in result["rejected"]
    )


def test_select_best_returns_none_when_infeasible():
    from everyngram.search import select_best

    rows = [
        _row("a", 1000, task=_task(0.3, 0.5, ood_ratio=1.9)),
        _row("b", 1000, task=None),
    ]

    result = select_best(rows, "task", ood_max_ratio=1.10)

    assert result["name"] is None
    assert result["reason"]
    assert len(result["rejected"]) == 2


def test_select_best_rejects_rows_missing_ood_reading_under_constraint():
    from everyngram.search import select_best

    rows = [_row("a", 1000, task=_task(0.3, 0.2, ood_ratio=None))]

    result = select_best(rows, "task", ood_max_ratio=1.10)

    assert result["name"] is None
    assert any("OOD" in item["reason"] for item in result["rejected"])


def test_select_best_rejects_unknown_objective():
    from everyngram.search import select_best

    with pytest.raises(ValueError, match="objective"):
        select_best([], "whatever")


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 8 for char in text]

    def get_vocab(self):
        return {str(index): index for index in range(8)}


class FakeModel:
    """下一位 = (当前位 + 1) % 8 的确定性模型,带缓存协议。"""

    def _logits(self, seq):
        import numpy as np

        out = np.zeros((len(seq), 8), dtype=np.float64)
        for index, token in enumerate(seq):
            out[index, (int(token) + 1) % 8] = 10.0
        return out

    def forward(self, token_ids, cache=None):
        tokens = [] if cache is None else list(cache)
        new = [int(t) for t in token_ids]
        return self._logits(tokens + new)[len(tokens) :], tokens + new

    def crop(self, cache, keep: int):
        return list(cache)[:keep]


def _write_corpus(root, rows: int = 12):
    root.mkdir(parents=True, exist_ok=True)
    text = "".join("| a | b |\n" for _ in range(rows))
    (root / "doc.txt").write_text(text, encoding="utf-8")


def _search_config(tmp_path, **overrides):
    from everyngram.config import SearchConfig

    corpus_a = tmp_path / "corpus_a"
    corpus_b = tmp_path / "corpus_b"
    _write_corpus(corpus_a)
    _write_corpus(corpus_b)
    ood_root = tmp_path / "ood"
    ood_root.mkdir(exist_ok=True)
    (ood_root / "sample.py").write_text(
        "def f():\n    return 1\n" * 200, encoding="utf-8"
    )
    base = dict(
        model="fake",
        out_dir=str(tmp_path / "out"),
        objective="task",
        ood_max_ratio=1.5,
        target_corpus=str(corpus_a),
        dev_scenarios=4,
        test_scenarios=4,
        context_len=16,
        tail_chars=200,
        horizon=4,
        candidates=(
            {"name": "a", "corpus": str(corpus_a), "budgets": (1000,)},
            {"name": "b", "corpus": str(corpus_b), "budgets": (1000,)},
        ),
        lambdas=(0.2,),
        gamma=2,
        max_new_tokens=4,
        ood_root=str(ood_root),
        ood_files=1,
        ood_max_tokens=400,
        ood_exclude=(),
        bootstrap=20,
    )
    base.update(overrides)
    return SearchConfig(**base)


def test_run_search_end_to_end_structures_and_discipline(tmp_path):
    from everyngram.memory import Memory
    from everyngram.search import run_search
    from everyngram.hf import tokenizer_fingerprint

    cfg = _search_config(tmp_path)
    tokenizer = FakeTokenizer()
    model = FakeModel()

    report = run_search(model, tokenizer, cfg)

    assert len(report["rows"]) == 2
    for row in report["rows"]:
        assert row["entries"] > 0
        assert "a_model" in row and "a_target" in row
        assert "feasible_reason" in row
    assert report["selected"] is not None
    assert report["test"] is not None
    assert report["test"]["name"] == report["selected"]["name"]
    table_path = tmp_path / "out" / "selected.ngmem"
    assert table_path.exists()
    loaded = Memory.load(table_path)
    assert loaded.fingerprint == tokenizer_fingerprint(tokenizer)

    # S4 确定性:同配置第二次运行给出相同选择与读数
    again = run_search(model, tokenizer, cfg)
    assert again["selected"]["name"] == report["selected"]["name"]
    assert again["rows"] == report["rows"]
    assert again["test"] == report["test"]


def test_run_search_screening_leaves_no_selection(tmp_path):
    from everyngram.search import run_search

    cfg = _search_config(tmp_path, min_a_target=2.0)

    report = run_search(FakeModel(), FakeTokenizer(), cfg)

    assert report["selected"] is None
    assert all("screened_out" in row for row in report["rows"])
    assert not (tmp_path / "out" / "selected.ngmem").exists()


def test_load_target_scenarios_honours_scenario_offset(tmp_path):
    from everyngram.config import SearchConfig
    from everyngram.search import _load_target_scenarios
    from everyngram.tasks import collect_table_prompts

    corpus = tmp_path / "corpus"
    _write_corpus(corpus, rows=24)
    tokenizer = FakeTokenizer()
    cfg = SearchConfig(
        model="fake",
        out_dir=str(tmp_path / "out"),
        objective="task",
        target_corpus=str(corpus),
        dev_scenarios=4,
        test_scenarios=4,
        scenario_offset=8,
        context_len=16,
        tail_chars=200,
        horizon=4,
    )

    dev, test = _load_target_scenarios(tokenizer, cfg)
    prompts = collect_table_prompts(
        sorted(corpus.rglob("*.txt")),
        tokenizer,
        n=16,
        context_len=16,
        tail_chars=200,
        horizon=4,
    )

    assert [s["prompt_ids"] for s in dev] == [
        item["prompt_ids"] for item in prompts[8:12]
    ]
    assert [s["prompt_ids"] for s in test] == [
        item["prompt_ids"] for item in prompts[12:16]
    ]


def test_run_search_fails_fast_when_ood_root_missing(tmp_path):
    from everyngram.search import run_search

    cfg = _search_config(tmp_path, ood_root=str(tmp_path / "missing_ood"))

    with pytest.raises(FileNotFoundError, match="OOD"):
        run_search(FakeModel(), FakeTokenizer(), cfg)
