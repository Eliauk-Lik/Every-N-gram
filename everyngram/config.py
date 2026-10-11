"""YAML 配置的加载与校验。字段错误一律在进入重计算之前抛出。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from everyngram.fuse import MODES


@dataclass(frozen=True)
class BuildConfig:
    model: str
    corpus: str
    out: str
    orders: tuple[int, ...] = (1, 2, 3)
    budget: int = 1_000_000
    continuation_len: int = 16


@dataclass(frozen=True)
class BenchConfig:
    model: str
    memory: str
    out: str
    corpus: str = ""
    ood_corpus: str = "data/ood.txt"
    max_files: int = 40
    lambdas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4)
    modes: tuple[str, ...] = ("arithmetic",)
    gammas: tuple[int, ...] = (1, 2, 4, 8)
    gamma: int = 4
    max_new_tokens: int = 32
    repeats: int = 3


@dataclass(frozen=True)
class SearchConfig:
    """eng search:在候选语料 × 预算之间按预注册规则为目标文档挑一张表。"""

    model: str
    out_dir: str
    dtype: str = "float32"
    objective: str = "task"
    ood_max_ratio: float | None = 1.10
    accept_length_min: float | None = None
    min_a_model: float | None = None
    min_a_target: float | None = None
    target_corpus: str = ""
    dev_scenarios: int = 32
    test_scenarios: int = 32
    scenario_offset: int = 0
    context_len: int = 128
    tail_chars: int = 2000
    horizon: int = 16
    candidates: tuple[dict[str, Any], ...] = ()
    lambdas: tuple[float, ...] = (0.1, 0.2)
    gamma: int = 8
    max_new_tokens: int = 32
    orders: tuple[int, ...] = (1, 2, 3)
    continuation_len: int = 16
    ood_root: str = ""
    ood_exclude: tuple[str, ...] = ("test",)
    ood_files: int = 4
    ood_max_tokens: int = 2000
    bootstrap: int = 200
    seed: int = 0


def _load(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置必须是映射: {path}")
    return payload


def _check_keys(payload: dict, allowed: set[str], required: set[str]) -> None:
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"缺少必需字段: {', '.join(missing)}")
    unknown = sorted(payload.keys() - allowed)
    if unknown:
        raise ValueError(f"未知字段: {', '.join(unknown)}")


def load_build_config(path: str | Path) -> BuildConfig:
    payload = _load(path)
    _check_keys(
        payload,
        allowed={"model", "corpus", "out", "orders", "budget", "continuation_len"},
        required={"model", "corpus", "out"},
    )
    return BuildConfig(
        model=str(payload["model"]),
        corpus=str(payload["corpus"]),
        out=str(payload["out"]),
        orders=tuple(int(x) for x in payload.get("orders", (1, 2, 3))),
        budget=int(payload.get("budget", 1_000_000)),
        continuation_len=int(payload.get("continuation_len", 16)),
    )


def load_bench_config(path: str | Path) -> BenchConfig:
    payload = _load(path)
    _check_keys(
        payload,
        allowed={
            "model", "memory", "out", "corpus", "ood_corpus", "max_files",
            "lambdas", "modes", "gammas", "gamma", "max_new_tokens", "repeats",
        },
        required={"model", "memory", "out"},
    )

    lambdas = tuple(float(x) for x in payload.get("lambdas", (0.0, 0.05, 0.1, 0.2, 0.4)))
    bad = [value for value in lambdas if not 0.0 <= value <= 1.0]
    if bad:
        raise ValueError(f"lambdas 必须在 [0, 1] 内: {bad}")

    modes = tuple(str(x) for x in payload.get("modes", ("arithmetic",)))
    bad_modes = [mode for mode in modes if mode not in MODES]
    if bad_modes:
        raise ValueError(f"modes 只能是 {MODES}: {bad_modes}")

    return BenchConfig(
        model=str(payload["model"]),
        memory=str(payload["memory"]),
        out=str(payload["out"]),
        corpus=str(payload.get("corpus", "")),
    ood_corpus=str(payload.get("ood_corpus", "data/ood.txt")),
        max_files=int(payload.get("max_files", 40)),
        lambdas=lambdas,
        modes=modes,
        gammas=tuple(int(x) for x in payload.get("gammas", (1, 2, 4, 8))),
        gamma=int(payload.get("gamma", 4)),
        max_new_tokens=int(payload.get("max_new_tokens", 32)),
        repeats=int(payload.get("repeats", 3)),
    )


_OBJECTIVES = ("task", "speed", "both")


def _optional_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return float(value)


def _check_mapping(
    payload: dict, field: str, allowed: set[str], required: set[str] = frozenset()
) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{field} 必须是映射")
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{field} 缺少必需字段: {', '.join(missing)}")
    unknown = sorted(payload.keys() - allowed)
    if unknown:
        raise ValueError(f"{field} 未知字段: {', '.join(unknown)}")
    return payload


def _parse_candidates(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("candidates 必须是非空列表")
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        entry = _check_mapping(
            item,
            f"candidates[{index}]",
            allowed={"name", "corpus", "budgets"},
            required={"name", "corpus", "budgets"},
        )
        budgets = tuple(int(b) for b in entry["budgets"])
        if not budgets or any(b <= 0 for b in budgets):
            raise ValueError(f"candidates[{index}].budgets 必须是正整数的非空列表")
        candidates.append(
            {
                "name": str(entry["name"]),
                "corpus": str(entry["corpus"]),
                "budgets": budgets,
            }
        )
    names = [candidate["name"] for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidates 的 name 必须唯一")
    return tuple(candidates)


def load_search_config(path: str | Path) -> SearchConfig:
    payload = _load(path)
    _check_keys(
        payload,
        allowed={
            "model", "out_dir", "dtype", "objective", "constraints", "screening",
            "target", "candidates", "lambdas", "gamma", "max_new_tokens",
            "orders", "continuation_len", "ood", "bootstrap", "seed",
        },
        required={"model", "out_dir", "target", "candidates"},
    )

    objective = str(payload.get("objective", "task"))
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective 只能是 {_OBJECTIVES}: {objective}")

    constraints = _check_mapping(
        payload.get("constraints", {}),
        "constraints",
        allowed={"ood_max_ratio", "accept_length_min"},
    )
    screening = _check_mapping(
        payload.get("screening", {}),
        "screening",
        allowed={"min_a_model", "min_a_target"},
    )
    target = _check_mapping(
        payload["target"],
        "target",
        allowed={
            "corpus", "dev_scenarios", "test_scenarios", "scenario_offset",
            "context_len", "tail_chars", "horizon",
        },
        required={"corpus"},
    )
    ood = _check_mapping(
        payload.get("ood", {}),
        "ood",
        allowed={"root", "exclude", "files", "max_tokens"},
        required={"root"},
    )

    lambdas = tuple(float(x) for x in payload.get("lambdas", (0.1, 0.2)))
    bad = [value for value in lambdas if not 0.0 <= value <= 1.0]
    if bad or not lambdas:
        raise ValueError(f"lambdas 必须是 [0, 1] 内的非空列表: {bad or lambdas}")

    dev_scenarios = int(target.get("dev_scenarios", 32))
    test_scenarios = int(target.get("test_scenarios", 32))
    if dev_scenarios <= 0 or test_scenarios <= 0:
        raise ValueError("target.dev_scenarios 与 target.test_scenarios 必须为正")

    return SearchConfig(
        model=str(payload["model"]),
        out_dir=str(payload["out_dir"]),
        dtype=str(payload.get("dtype", "float32")),
        objective=objective,
        ood_max_ratio=_optional_float(constraints.get("ood_max_ratio", 1.10), "ood_max_ratio"),
        accept_length_min=_optional_float(
            constraints.get("accept_length_min"), "accept_length_min"
        ),
        min_a_model=_optional_float(screening.get("min_a_model"), "min_a_model"),
        min_a_target=_optional_float(screening.get("min_a_target"), "min_a_target"),
        target_corpus=str(target["corpus"]),
        dev_scenarios=dev_scenarios,
        test_scenarios=test_scenarios,
        scenario_offset=int(target.get("scenario_offset", 0)),
        context_len=int(target.get("context_len", 128)),
        tail_chars=int(target.get("tail_chars", 2000)),
        horizon=int(target.get("horizon", 16)),
        candidates=_parse_candidates(payload["candidates"]),
        lambdas=lambdas,
        gamma=int(payload.get("gamma", 8)),
        max_new_tokens=int(payload.get("max_new_tokens", 32)),
        orders=tuple(int(x) for x in payload.get("orders", (1, 2, 3))),
        continuation_len=int(payload.get("continuation_len", 16)),
        ood_root=str(ood["root"]),
        ood_exclude=tuple(str(x) for x in ood.get("exclude", ("test",))),
        ood_files=int(ood.get("files", 4)),
        ood_max_tokens=int(ood.get("max_tokens", 2000)),
        bootstrap=int(payload.get("bootstrap", 200)),
        seed=int(payload.get("seed", 0)),
    )
