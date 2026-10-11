import pytest

from everyngram.config import (
    load_bench_config,
    load_build_config,
    load_search_config,
)


def test_load_build_config_parses_yaml(tmp_path):
    path = tmp_path / "build.yaml"
    path.write_text(
        "model: data/models/Qwen3-0.6B\n"
        "corpus: data/project-docs\n"
        "out: reports/inc3.ngmem\n"
        "orders: [1, 2, 3]\n"
        "budget: 5000\n",
        encoding="utf-8",
    )

    cfg = load_build_config(path)

    assert cfg.model == "data/models/Qwen3-0.6B"
    assert cfg.orders == (1, 2, 3)
    assert cfg.budget == 5000
    assert cfg.continuation_len == 16


def test_load_bench_config_parses_lambdas(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text(
        "model: data/models/Qwen3-0.6B\n"
        "memory: reports/inc3.ngmem\n"
        "out: reports/inc3\n"
        "lambdas: [0.0, 0.1, 0.4]\n"
        "modes: [arithmetic, geometric]\n",
        encoding="utf-8",
    )

    cfg = load_bench_config(path)

    assert cfg.lambdas == (0.0, 0.1, 0.4)
    assert cfg.modes == ("arithmetic", "geometric")
    assert cfg.gamma == 4


def test_missing_field_raises_with_field_name(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model: m\n", encoding="utf-8")

    with pytest.raises(ValueError, match="corpus"):
        load_build_config(path)


def test_unknown_field_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model: m\ncorpus: c\nout: o\nnope: 1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="nope"):
        load_build_config(path)


def test_load_search_config_parses_nested_sections(tmp_path):
    path = tmp_path / "search.yaml"
    path.write_text(
        "model: data/models/Qwen3-0.6B\n"
        "out_dir: reports/search_demo\n"
        "objective: task\n"
        "constraints: {ood_max_ratio: 1.10}\n"
        "screening: {min_a_target: 0.5}\n"
        "target:\n"
        "  corpus: data/project-docs\n"
        "  dev_scenarios: 16\n"
        "  test_scenarios: 16\n"
        "candidates:\n"
        "  - {name: c1, corpus: data/project-docs, budgets: [8000, 20000]}\n"
        "lambdas: [0.1, 0.2]\n"
        "ood: {root: /tmp/lib, files: 2, max_tokens: 500}\n",
        encoding="utf-8",
    )

    cfg = load_search_config(path)

    assert cfg.objective == "task"
    assert cfg.ood_max_ratio == 1.10
    assert cfg.min_a_target == 0.5
    assert cfg.min_a_model is None
    assert cfg.target_corpus == "data/project-docs"
    assert cfg.dev_scenarios == 16 and cfg.test_scenarios == 16
    assert cfg.candidates == (
        {"name": "c1", "corpus": "data/project-docs", "budgets": (8000, 20000)},
    )
    assert cfg.lambdas == (0.1, 0.2)
    assert cfg.ood_files == 2 and cfg.ood_max_tokens == 500


def test_load_search_config_rejects_unknown_objective(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model: m\nout_dir: o\nobjective: whatever\n"
        "target: {corpus: c}\n"
        "candidates: [{name: a, corpus: c1, budgets: [100]}]\n"
        "ood: {root: /tmp/lib}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="objective"):
        load_search_config(path)


def test_load_search_config_rejects_empty_candidates(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model: m\nout_dir: o\ntarget: {corpus: c}\ncandidates: []\n"
        "ood: {root: /tmp/lib}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidates"):
        load_search_config(path)


def test_load_search_config_rejects_bad_lambdas(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model: m\nout_dir: o\ntarget: {corpus: c}\n"
        "candidates: [{name: a, corpus: c1, budgets: [100]}]\n"
        "lambdas: [0.5, 1.5]\n"
        "ood: {root: /tmp/lib}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lambdas"):
        load_search_config(path)


def test_lambda_out_of_range_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model: m\nmemory: m.ngmem\nout: o\nlambdas: [1.5]\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="lambdas"):
        load_bench_config(path)


def test_unknown_mode_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model: m\nmemory: m.ngmem\nout: o\nmodes: [magic]\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="modes"):
        load_bench_config(path)


def test_bench_config_declares_ood_corpus_with_default(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text("model: m\nmemory: m.ngmem\nout: o\n", encoding="utf-8")

    assert load_bench_config(path).ood_corpus == "data/ood.txt"


def test_bench_config_accepts_explicit_ood_corpus(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text(
        "model: m\nmemory: m.ngmem\nout: o\nood_corpus: docs/ood.md\n",
        encoding="utf-8",
    )

    assert load_bench_config(path).ood_corpus == "docs/ood.md"
