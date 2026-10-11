import pytest

from everyngram import cli
import subprocess
import sys


def test_main_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])


def test_build_calls_run_build_with_parsed_config(tmp_path, monkeypatch):
    path = tmp_path / "build.yaml"
    path.write_text(
        "model: m\ncorpus: c\nout: o\n", encoding="utf-8"
    )
    captured = {}

    def fake_run_build(cfg):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(cli, "run_build", fake_run_build)

    assert cli.main(["build", "--config", str(path)]) == 0
    assert captured["cfg"].model == "m"


def test_bench_calls_run_bench_with_parsed_config(tmp_path, monkeypatch):
    path = tmp_path / "bench.yaml"
    path.write_text(
        "model: m\nmemory: x.ngmem\nout: o\n", encoding="utf-8"
    )
    captured = {}

    def fake_run_bench(cfg):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(cli, "run_bench", fake_run_bench)

    assert cli.main(["bench", "--config", str(path)]) == 0
    assert captured["cfg"].memory == "x.ngmem"


def test_module_entry_point_prints_help():
    result = subprocess.run(
        [sys.executable, "-m", "everyngram.cli", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "build" in result.stdout and "bench" in result.stdout
    assert "search" in result.stdout


def test_search_calls_run_search_command_with_parsed_config(tmp_path, monkeypatch):
    path = tmp_path / "search.yaml"
    path.write_text(
        "model: m\n"
        "out_dir: reports/search_demo\n"
        "objective: task\n"
        "target: {corpus: data/project-docs}\n"
        "candidates:\n"
        "  - {name: c1, corpus: data/project-docs, budgets: [8000]}\n"
        "ood: {root: /tmp/lib}\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run_search(cfg):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(cli, "run_search_command", fake_run_search)

    assert cli.main(["search", "--config", str(path)]) == 0
    assert captured["cfg"].objective == "task"
    assert captured["cfg"].candidates[0]["name"] == "c1"


def test_resolve_ood_corpus_returns_absolute_existing_path(tmp_path):
    from everyngram.cli import resolve_ood_corpus

    target = tmp_path / "corpus.md"
    target.write_text("hello", encoding="utf-8")

    resolved = resolve_ood_corpus(str(target))

    assert resolved.is_absolute()
    assert resolved == target.resolve()


def test_resolve_ood_corpus_reports_absolute_path_when_missing(tmp_path):
    from everyngram.cli import resolve_ood_corpus

    missing = tmp_path / "nope.md"

    with pytest.raises(FileNotFoundError) as info:
        resolve_ood_corpus(str(missing))

    assert str(missing.resolve()) in str(info.value)


def test_run_bench_degrades_gamma_and_reports_the_value_it_used(tmp_path, monkeypatch):
    """显存不足降级时,报告里写的 gamma 必须等于实际传入的值(审计 P8)。"""
    import json

    import everyngram.bench as bench_mod
    import everyngram.hf as hf_mod
    from everyngram.memory import Memory

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 16 for ch in text]

    memory = Memory.build([1, 2, 3, 4] * 10, fingerprint="fp", orders=(1,), budget=50)
    memory_path = tmp_path / "m.ngmem"
    memory.save(memory_path)
    ood = tmp_path / "ood.md"
    ood.write_text("hello world", encoding="utf-8")

    cfg = cli.BenchConfig(
        model="m",
        memory=str(memory_path),
        out=str(tmp_path / "out"),
        corpus="",
        ood_corpus=str(ood),
        gamma=8,
        gammas=(1, 2, 4, 8),
        max_new_tokens=2,
        repeats=1,
    )

    monkeypatch.setattr(hf_mod, "load_tokenizer", lambda name: FakeTokenizer())
    monkeypatch.setattr(hf_mod, "tokenizer_fingerprint", lambda tokenizer: "fp")
    monkeypatch.setattr(hf_mod, "HFModel", lambda name: object())

    gammas_used: list[int] = []

    def fake_scan(model, memory, **kwargs):
        gammas_used.append(kwargs["gamma"])
        if len(gammas_used) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return {
            "baseline": {"ms_per_token": 1.0, "in_domain": 0.0, "ood_perplexity": 1.0},
            "rows": [],
        }

    monkeypatch.setattr(bench_mod, "run_fusion_scan", fake_scan)
    monkeypatch.setattr(bench_mod, "plot_tradeoff", lambda scan, path: None)

    assert cli.run_bench(cfg) == 0
    assert gammas_used == [8, 4]
    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert report["degraded"] == ["显存不足:gamma 从 8 降到 4"]
    assert report["ood_corpus"] == str(ood.resolve())
