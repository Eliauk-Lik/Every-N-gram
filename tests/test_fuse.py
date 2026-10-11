import numpy as np
import pytest

from everyngram.fuse import (
    FusionSpec,
    fuse_probs,
    fused_next_probs,
    memory_dist,
)
from everyngram.memory import Entry, Memory


def test_fusion_spec_rejects_bad_values():
    with pytest.raises(ValueError):
        FusionSpec(lam=1.5)
    with pytest.raises(ValueError):
        FusionSpec(lam=0.1, mode="nope")


def test_memory_dist_normalizes_top_counts():
    entry = Entry(count=4, next_top=((2, 3), (1, 1)), continuation=(2,))

    q = memory_dist(entry, vocab_size=4)

    assert np.allclose(q, [0.0, 0.25, 0.75, 0.0])


def test_memory_dist_returns_zeros_for_missing_or_invalid():
    entry = Entry(count=2, next_top=((9, 5),), continuation=())

    assert np.allclose(memory_dist(entry, vocab_size=4), np.zeros(4))
    assert np.allclose(memory_dist(None, vocab_size=4), np.zeros(4))


def test_lambda_zero_is_identity_for_both_modes():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.0, 0.5, 0.5])

    for mode in ("arithmetic", "geometric"):
        assert np.allclose(fuse_probs(p, q, lam=0.0, mode=mode), p)


def test_arithmetic_support_is_union():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.0, 0.5, 0.5])

    mixed = fuse_probs(p, q, lam=0.2, mode="arithmetic")

    assert np.allclose(mixed, [0.48, 0.34, 0.18])
    assert abs(mixed.sum() - 1.0) < 1e-12


def test_geometric_support_is_intersection():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.0, 0.5, 0.5])

    mixed = fuse_probs(p, q, lam=0.2, mode="geometric")

    assert mixed[0] == 0.0
    assert mixed[1] > 0.0 and mixed[2] > 0.0
    assert abs(mixed.sum() - 1.0) < 1e-12


def test_lambda_one_returns_memory_distribution():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.0, 0.5, 0.5])

    assert np.allclose(fuse_probs(p, q, lam=1.0, mode="arithmetic"), q)
    assert np.allclose(fuse_probs(p, q, lam=1.0, mode="geometric"), q)


def test_fuse_returns_p_copy_when_q_is_empty():
    p = np.array([0.6, 0.4])

    mixed = fuse_probs(p, np.zeros(2), lam=0.5, mode="arithmetic")

    assert np.allclose(mixed, p)
    assert mixed is not p


def test_fused_next_probs_without_memory_is_softmax():
    logits = np.array([0.0, 1.0])

    probs = fused_next_probs(logits, [0], fusion=None)

    assert abs(probs.sum() - 1.0) < 1e-12
    assert probs[1] > probs[0]


def test_fused_next_probs_applies_memory_distribution():
    memory = Memory.build([0, 9] * 20, fingerprint="fp", orders=(1,), budget=100)
    spec = FusionSpec(lam=1.0, memory=memory)
    logits = np.zeros(16)
    logits[3] = 5.0

    probs = fused_next_probs(logits, [0], spec)

    assert int(np.argmax(probs)) == 9


def test_gate_blocks_low_order_and_low_share():
    from everyngram.fuse import Gate
    from everyngram.memory import Entry, Memory

    entry = Entry(count=10, next_top=((1, 6), (2, 4)), continuation=(1,))
    memory = Memory(
        fingerprint="fp",
        orders=(1, 2),
        budget=2,
        continuation_len=1,
        next_top_k=2,
        entries={(1,): entry, (5, 1): entry},
    )

    assert Gate(min_order=2).allows((1,), entry, 0.1) is False
    assert Gate(min_order=2).allows((5, 1), entry, 0.1) is True
    assert Gate(min_share=0.7).allows((1,), entry, 0.1) is False
    assert Gate(min_share=0.5).allows((1,), entry, 0.1) is True
    assert Gate(max_top_prob=0.5).allows((1,), entry, 0.9) is False
    assert Gate(max_top_prob=0.9).allows((1,), entry, 0.9) is True
    assert Gate(max_top_prob=1.0).allows((1,), entry, 1.0) is True
    assert memory.lookup([5, 1])[0] == (5, 1)


def test_gate_returns_model_distribution_when_blocked():
    from everyngram.fuse import Gate

    memory = Memory.build([0, 9] * 20, fingerprint="fp", orders=(1,), budget=100)
    logits = np.zeros(16)
    logits[3] = 5.0

    blocked = fused_next_probs(
        logits, [0], FusionSpec(lam=1.0, memory=memory, gate=Gate(min_order=2))
    )
    allowed = fused_next_probs(
        logits, [0], FusionSpec(lam=1.0, memory=memory, gate=Gate(min_order=1))
    )

    assert int(np.argmax(blocked)) == 3      # 门槛挡回 p_θ
    assert int(np.argmax(allowed)) == 9      # 通过门槛时按记忆融合


def test_gate_rejects_bad_values():
    from everyngram.fuse import Gate

    with pytest.raises(ValueError):
        Gate(min_order=0)
    with pytest.raises(ValueError):
        Gate(min_share=1.5)
    with pytest.raises(ValueError):
        Gate(max_top_prob=0.0)
