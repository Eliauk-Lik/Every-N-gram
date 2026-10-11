import numpy as np
import pytest

from everyngram.memory import Memory
from everyngram.tasks import EvalCase


class CountingCachedModel:
    """下一个 token 恒为 (当前 token + 1) % 16。"""

    def forward(self, token_ids, cache=None):
        tokens = [] if cache is None else list(cache)
        new = [int(t) for t in token_ids]
        out = np.zeros((len(tokens) + len(new), 16), dtype=np.float64)
        for i, token in enumerate(tokens + new):
            out[i, (int(token) + 1) % 16] = 1.0
        return out[len(tokens) :], tokens + new


def _case(prompt, reference, origin=""):
    return EvalCase(
        prompt_ids=tuple(prompt), reference_ids=tuple(reference), origin=origin
    )


def test_usage_separates_quality_keys_from_speed_keys():
    from everyngram.budget import score_entry_usage
    from everyngram.fuse import Gate

    # 键 (1,) 的续写是 9(模型不会 → quality 正、accepted 0);
    # 键 (4,) 首次续写是 5(被接受),但 q 把 6/7 的质量摊到 7 上,
    # 真值 5 的似然反而被稀释 → quality 负、accepted 1。
    corpus = [1, 9, 1, 9, 1, 9, 4, 5, 4, 7, 4, 7, 4, 7, 4, 7, 4, 7, 4, 7]
    memory = Memory.build(corpus, fingerprint="fp", orders=(1,), budget=100)
    cases = (_case([1], [9]), _case([4], [5]))

    usage = score_entry_usage(
        CountingCachedModel(), memory, cases, lam=0.5, gate=Gate(min_order=1)
    )

    assert usage["positions"] == 2
    assert usage["hits"] == 2
    assert usage["per_key"][(1,)]["queries"] == 1
    assert usage["per_key"][(4,)]["queries"] == 1
    assert usage["per_key"][(1,)]["quality"] > 0
    assert usage["per_key"][(4,)]["quality"] < 0
    assert usage["per_key"][(1,)]["accepted"] == 0
    assert usage["per_key"][(4,)]["accepted"] == 1


def test_usage_gate_blocks_quality_but_not_queries():
    from everyngram.budget import score_entry_usage
    from everyngram.fuse import Gate

    corpus = [1, 9, 1, 9, 1, 2, 4, 5]
    memory = Memory.build(corpus, fingerprint="fp", orders=(1,), budget=100)
    cases = (_case([1], [9]),)

    usage = score_entry_usage(
        CountingCachedModel(),
        memory,
        cases,
        lam=0.5,
        gate=Gate(min_order=1, min_share=0.99, max_top_prob=0.01),
    )

    row = usage["per_key"][(1,)]
    assert row["queries"] == 1
    assert row["quality"] == 0.0


def test_usage_reports_miss_positions():
    from everyngram.budget import score_entry_usage

    memory = Memory.build([1, 2, 1, 2], fingerprint="fp", orders=(2,), budget=100)
    cases = (_case([7, 7], [3]),)

    usage = score_entry_usage(CountingCachedModel(), memory, cases, lam=0.1)

    assert usage["positions"] == 1
    assert usage["hits"] == 0
    assert usage["per_key"] == {}


def test_usage_empty_reference_is_skipped():
    from everyngram.budget import score_entry_usage

    memory = Memory.build([1, 2, 1, 2], fingerprint="fp", orders=(1,), budget=100)

    usage = score_entry_usage(
        CountingCachedModel(), memory, (_case([1], []),), lam=0.1
    )

    assert usage["positions"] == 0
    assert usage["hits"] == 0


def test_usage_matches_fused_next_probs_on_a_position():
    from everyngram.budget import score_entry_usage
    from everyngram.fuse import Gate, fuse_probs, memory_dist, softmax

    corpus = [1, 9, 1, 9, 1, 9]
    memory = Memory.build(corpus, fingerprint="fp", orders=(1,), budget=100)
    cases = (_case([1], [9]),)
    model = CountingCachedModel()
    logits, _ = model.forward([1])
    p = softmax(logits[0])
    entry = memory.entries[(1,)]
    fused = fuse_probs(p, memory_dist(entry, p.shape[0]), 0.5, "arithmetic")
    expected = float(-np.log(p[9])) - float(-np.log(fused[9]))

    usage = score_entry_usage(model, memory, cases, lam=0.5, gate=Gate(min_order=1))

    assert usage["per_key"][(1,)]["quality"] == pytest.approx(expected, abs=1e-12)


def _usage(memory, rows):
    return {
        "per_key": {
            key: {"queries": q, "quality": qual, "accepted": acc}
            for key, (q, qual, acc) in rows.items()
        },
        "positions": 0,
        "hits": 0,
    }


def test_freq_strategy_matches_build_truncation():
    from everyngram.budget import select_keys

    corpus = [1, 2, 3, 1, 2, 3, 1, 2, 4]
    full = Memory.build(corpus, fingerprint="fp", orders=(1, 2, 3), budget=100)
    direct = Memory.build(corpus, fingerprint="fp", orders=(1, 2, 3), budget=4)

    keys = select_keys(full, _usage(full, {}), strategy="freq", budget=4)

    assert set(keys) == set(direct.entries)


def test_query_strategy_prefers_queried_keys():
    from everyngram.budget import select_keys

    corpus = [1, 2, 1, 2, 1, 2, 1, 2]
    memory = Memory.build(corpus, fingerprint="fp", orders=(1, 2), budget=100)
    usage = _usage(memory, {(1,): (2, 0.0, 0), (2,): (9, 0.0, 0)})

    keys = select_keys(memory, usage, strategy="query", budget=1)

    assert keys == ((2,),)


def test_quality_and_speed_strategies_pick_their_own_tops():
    from everyngram.budget import select_keys

    corpus = [1, 2, 1, 2, 1, 2, 1, 2]
    memory = Memory.build(corpus, fingerprint="fp", orders=(1, 2), budget=100)
    usage = _usage(memory, {(1,): (1, 0.5, 0), (2,): (1, -0.5, 3)})

    assert select_keys(memory, usage, strategy="quality", budget=1) == ((1,),)
    assert select_keys(memory, usage, strategy="speed", budget=1) == ((2,),)


def test_hybrid_unions_half_quality_half_speed_and_fills():
    from everyngram.budget import select_keys

    corpus = [1, 2, 1, 2, 1, 2, 1, 2]
    memory = Memory.build(corpus, fingerprint="fp", orders=(1, 2), budget=100)
    usage = _usage(memory, {(1,): (1, 0.9, 0), (2,): (1, -0.9, 5)})

    keys = select_keys(memory, usage, strategy="hybrid", budget=2)

    assert set(keys) == {(1,), (2,)}


def test_hybrid_fills_to_budget_when_top_halves_overlap():
    from everyngram.budget import select_keys

    corpus = [1, 2, 3, 1, 2, 3, 1, 2, 4]
    memory = Memory.build(corpus, fingerprint="fp", orders=(1, 2, 3), budget=100)
    usage = _usage(memory, {(1,): (5, 1.0, 5)})

    keys = select_keys(memory, usage, strategy="hybrid", budget=4)

    assert len(keys) == 4
    assert keys[0] == (1,)


def test_select_keys_caps_at_table_size_and_validates_inputs():
    from everyngram.budget import select_keys

    memory = Memory.build([1, 2, 1, 2], fingerprint="fp", orders=(1,), budget=100)
    usage = _usage(memory, {})

    with pytest.raises(ValueError):
        select_keys(memory, usage, strategy="magic", budget=1)
    with pytest.raises(ValueError):
        select_keys(memory, usage, strategy="freq", budget=-1)
    assert select_keys(memory, usage, strategy="freq", budget=0) == ()
    assert len(select_keys(memory, usage, strategy="freq", budget=99)) == len(
        memory.entries
    )


def test_select_keys_is_deterministic_for_tied_scores():
    from everyngram.budget import select_keys

    corpus = [1, 2, 3, 1, 2, 3, 1, 2, 4]
    memory = Memory.build(corpus, fingerprint="fp", orders=(1, 2, 3), budget=100)
    usage = _usage(memory, {(1,): (1, 0.0, 0), (2,): (1, 0.0, 0)})

    first = select_keys(memory, usage, strategy="query", budget=3)
    second = select_keys(memory, usage, strategy="query", budget=3)

    assert first == second
