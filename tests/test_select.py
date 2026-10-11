import numpy as np
import pytest

from everyngram.memory import Memory
from everyngram.select import collect_contexts, score_contexts, select_pools


class FakeTokenizer:
    """每个字符映射到 0 到 15 的 token id。"""

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 16 for ch in text]


class CountingCachedModel:
    """下一个 token 恒为 (当前 token + 1) % 16。"""

    def forward(self, token_ids, cache=None):
        out = np.zeros((len(token_ids), 16), dtype=np.float64)
        for i, token in enumerate(token_ids):
            out[i, (int(token) + 1) % 16] = 1.0
        return out, list(cache or []) + [int(t) for t in token_ids]

    def crop(self, cache, keep: int):
        return list(cache)[:keep]


def test_collect_contexts_returns_fixed_length_contexts(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("x = 1\n" * 200, encoding="utf-8")

    contexts = collect_contexts([path], FakeTokenizer(), n=5, context_len=8, seed=0)

    assert len(contexts) == 5
    for prompt, true_next in contexts:
        assert len(prompt) == 8
        assert isinstance(true_next, int)


def test_collect_contexts_caps_at_available_positions(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("abc", encoding="utf-8")

    contexts = collect_contexts([path], FakeTokenizer(), n=50, context_len=8, seed=0)

    assert len(contexts) == 0


def test_speed_and_quality_scores_point_in_opposite_directions():
    # 模型预测 +1。键 (1,) 主要续写 9(模型不会),键 (4,) 只续写 5(模型也会)
    corpus = [1, 9, 1, 9, 1, 9, 1, 2, 4, 5]
    memory = Memory.build(corpus, fingerprint="fp", orders=(1,), budget=100)
    contexts = (((1,), 9), ((4,), 5))

    scored = score_contexts(CountingCachedModel(), memory, contexts)

    knows_truth = scored[0]      # 真值 9:模型错、记忆对 → 质量分高、速度分低
    agrees = scored[1]           # 真值 5:模型本来就对 → 速度分高、质量分负
    assert knows_truth["quality_score"] > agrees["quality_score"]
    assert knows_truth["speed_score"] < agrees["speed_score"]
    assert knows_truth["draft_first"] == 9
    assert agrees["draft_first"] == 5


def test_select_pools_reports_overlap_and_caps_size():
    scored = [
        {
            "key": (i,),
            "true_next": (i + 1) % 16,
            "speed_score": float(i) / 16,
            "quality_score": float(16 - i) / 16,
        }
        for i in range(8)
    ]

    pools = select_pools(scored, k=4)

    assert len(pools["speed_pool"]) <= 4
    assert len(pools["quality_pool"]) <= 4
    assert 0 <= pools["overlap"] <= 1
    assert pools["k_effective"] == 4


def test_select_pools_rejects_empty_candidate_set():
    with pytest.raises(ValueError):
        select_pools([], k=4)


def test_effective_k_shrinks_when_candidates_are_scarce():
    from everyngram.select import effective_k

    scored = [
        {"key": (i,), "true_next": 0, "speed_score": 0.5, "quality_score": 0.5}
        for i in range(10)
    ]
    scored.append({"key": None, "true_next": 0, "speed_score": None,
                   "quality_score": None})

    assert effective_k(scored, 256) == 5
    assert effective_k(scored, 3) == 3
    assert effective_k(scored, 0) == 1


def test_select_pools_reports_overlap_count_and_disjoint_pools():
    scored = [
        {"key": (i,), "true_next": 0,
         "speed_score": float(i), "quality_score": float(-i)}
        for i in range(8)
    ]

    pools = select_pools(scored, k=2)

    assert pools["overlap_count"] == 0
    assert pools["overlap"] == 0.0
    assert {item["key"] for item in pools["speed_pool"]} == {(7,), (6,)}
    assert {item["key"] for item in pools["quality_pool"]} == {(0,), (1,)}


def test_overlap_curve_reports_zero_when_rankings_are_opposite():
    from everyngram.select import overlap_curve

    scored = [
        {"key": (i,), "true_next": 0, "speed_score": float(i),
         "quality_score": float(-i)}
        for i in range(8)
    ]

    curve = overlap_curve(scored, ks=(2, 4, 8))

    assert curve["ks"] == [2, 4, 8]
    assert curve["overlap"] == [0.0, 0.0, 1.0]
    assert curve["intersection"] == [0, 0, 8]


def test_select_gate_picks_best_dev_config_and_respects_coverage():
    from everyngram.fuse import Gate
    from everyngram.select import select_gate
    from everyngram.tasks import EvalCase

    # 键 (1,) 续写 9(模型不会),键 (4,) 续写 5(模型也会)
    memory = Memory.build(
        [1, 9] * 12 + [4, 5] * 12, fingerprint="fp", orders=(1,), budget=100
    )
    cases = (
        EvalCase(prompt_ids=(1,), reference_ids=(9,)),
        EvalCase(prompt_ids=(4,), reference_ids=(5,)),
    )
    gates = (
        Gate(min_order=1),
        Gate(min_order=2),          # 1-gram 表里永不通过 → 覆盖率 0
    )

    out = select_gate(
        CountingCachedModel(), memory, cases, gates=gates, lambdas=(1.0,),
        min_coverage=0.25,
    )

    assert out["n_positions"] == 2
    assert out["best"]["gate"] == {
        "min_order": 1, "min_share": 0.0, "max_top_prob": 1.0,
    }
    assert out["best"]["coverage"] == 1.0
    assert out["best"]["dev_gain"] > 0
    coverages = {row["gate"]["min_order"]: row["coverage"] for row in out["rows"]}
    assert coverages[2] == 0.0


def test_select_gate_returns_none_when_coverage_too_low():
    from everyngram.fuse import Gate
    from everyngram.select import select_gate
    from everyngram.tasks import EvalCase

    memory = Memory.build([1, 9] * 12, fingerprint="fp", orders=(1,), budget=100)
    cases = (EvalCase(prompt_ids=(1,), reference_ids=(9,)),)

    out = select_gate(
        CountingCachedModel(), memory, cases, gates=(Gate(min_order=3),),
        lambdas=(1.0,), min_coverage=0.25,
    )

    assert out["best"] is None
    with pytest.raises(ValueError):
        select_gate(CountingCachedModel(), memory, cases, gates=(), lambdas=(1.0,))
