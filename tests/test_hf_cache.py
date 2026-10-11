import os

import numpy as np
import pytest

from everyngram.hf import HFModel

pytestmark = pytest.mark.skipif(
    os.environ.get("EVERYNGRAM_HF_TEST") != "1",
    reason="需要真实模型,设置 EVERYNGRAM_HF_TEST=1 后才跑",
)

MODEL = "data/models/Qwen3-0.6B"


def test_cached_forward_matches_stateless():
    model = HFModel(MODEL)
    ids = [1, 2, 3, 4]

    logits, cache = model.forward(ids)
    assert np.allclose(logits, model.next_logits(ids), atol=1e-4)

    more, cache = model.forward([5], cache)
    assert np.allclose(more, model.next_logits(ids + [5])[-1:], atol=1e-4)


def test_crop_then_forward_matches_fresh_prefix():
    model = HFModel(MODEL)

    _, cache = model.forward([1, 2, 3, 4])
    cropped = model.crop(cache, 2)

    logits, _ = model.forward([9], cropped)
    assert np.allclose(logits, model.next_logits([1, 2, 9])[-1:], atol=1e-4)


def test_real_model_three_way_equivalence():
    from everyngram.decode import greedy_generate
    from everyngram.memory import Memory

    model = HFModel(MODEL)
    prompt = [100, 200, 300]

    plain, _ = greedy_generate(model, prompt, max_new_tokens=8)
    cached, _ = greedy_generate(model, prompt, max_new_tokens=8, use_cache=True)
    assert cached == plain

    memory = Memory.build(
        list(range(64)) * 3,
        fingerprint="fp",
        orders=(1, 2, 3),
        budget=1000,
        continuation_len=8,
    )
    spec, stats = greedy_generate(
        model, prompt, max_new_tokens=8, memory=memory, gamma=4, use_cache=True
    )
    assert spec == plain
    assert stats.lookups > 0
