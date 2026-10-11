import numpy as np
import pytest

from everyngram.decode import CachedModel, greedy_generate


class FakeCachedModel:
    """把 CountingModel 的语义搬到缓存接口上:cache 内部是已处理 token 列表。"""

    def __init__(self, vocab: int = 16) -> None:
        self.vocab = vocab
        self.calls = 0
        self.forwarded_tokens = 0
        self.last_cache = []

    def _logits(self, token_ids):
        out = np.zeros((len(token_ids), self.vocab), dtype=np.float64)
        for i, token in enumerate(token_ids):
            out[i, (int(token) + 1) % self.vocab] = 1.0
        return out

    def forward(self, token_ids, cache=None):
        tokens = [] if cache is None else list(cache)
        new = [int(t) for t in token_ids]
        self.calls += 1
        self.forwarded_tokens += len(new)
        logits = self._logits(tokens + new)[len(tokens):]
        self.last_cache = tokens + new
        return logits, self.last_cache

    def crop(self, cache, keep: int):
        self.last_cache = list(cache)[:keep]
        return self.last_cache


def test_fake_model_implements_protocol():
    assert isinstance(FakeCachedModel(), CachedModel)


def test_cached_forward_returns_only_new_positions():
    model = FakeCachedModel()
    logits, cache = model.forward([0, 1, 2])

    assert logits.shape == (3, 16)
    assert cache == [0, 1, 2]

    more, cache = model.forward([3], cache)
    assert more.shape == (1, 16)
    assert cache == [0, 1, 2, 3]
    assert int(np.argmax(more[-1])) == 4


def test_crop_then_forward_equals_fresh_forward():
    model = FakeCachedModel()
    _, cache = model.forward([0, 1, 2, 3])
    cache = model.crop(cache, 2)

    logits, cache = model.forward([7], cache)
    fresh, _ = FakeCachedModel().forward([0, 1, 7])

    assert np.array_equal(logits, fresh[-1:])
    assert cache == [0, 1, 7]


class CountingModel:
    """下一个 token 恒为 (当前 token + 1) % vocab 的无状态模型。"""

    def __init__(self, vocab: int = 16) -> None:
        self.vocab = vocab

    def next_logits(self, token_ids):
        out = np.zeros((len(token_ids), self.vocab), dtype=np.float64)
        for i, token in enumerate(token_ids):
            out[i, (int(token) + 1) % self.vocab] = 1.0
        return out


def test_cached_plain_greedy_matches_stateless():
    a, sa = greedy_generate(CountingModel(), [0], max_new_tokens=6)
    b, sb = greedy_generate(FakeCachedModel(), [0], max_new_tokens=6, use_cache=True)

    assert a == b
    assert (sa.steps, sa.produced) == (sb.steps, sb.produced) == (6, 6)


def test_cached_plain_greedy_forwards_each_token_once():
    model = FakeCachedModel()
    greedy_generate(model, [0, 1], max_new_tokens=5, use_cache=True)

    # 1 次 prefill(2 个 token)+ 4 次单 token 前向 = 6 个 token,无重算
    assert model.calls == 5
    assert model.forwarded_tokens == 6


def test_cached_speculative_matches_stateless():
    from everyngram.memory import Memory

    cycle = [i % 16 for i in range(200)]
    memory = Memory.build(
        cycle, fingerprint="fp", orders=(1, 2, 3), budget=1000, continuation_len=16
    )

    plain, _ = greedy_generate(CountingModel(), [0], max_new_tokens=12)
    stateless, s1 = greedy_generate(
        CountingModel(), [0], max_new_tokens=12, memory=memory, gamma=4
    )
    cached, s2 = greedy_generate(
        FakeCachedModel(), [0], max_new_tokens=12, memory=memory, gamma=4, use_cache=True
    )

    assert cached == stateless == plain
    assert s2.as_dict() == s1.as_dict()


def test_cached_speculative_merges_bonus_into_next_draft():
    from everyngram.memory import Memory

    cycle = [i % 16 for i in range(200)]
    memory = Memory.build(
        cycle, fingerprint="fp", orders=(1, 2, 3), budget=1000, continuation_len=16
    )
    model = FakeCachedModel()

    out, stats = greedy_generate(
        model, [0], max_new_tokens=12, memory=memory, gamma=4, use_cache=True
    )
    plain, _ = greedy_generate(CountingModel(), [0], max_new_tokens=12)

    assert out == plain
    # 合并前:prefill + 每轮(验证 + bonus)= 1 + 3 + 2 = 6 次前向;
    # 合并后:prefill + 首轮验证 + 两轮"修正+下一轮草稿" = 4 次。
    assert model.calls == 4
    assert stats.drafted == 12


def test_cached_speculative_fusion_lookup_uses_full_context():
    from everyngram.fuse import FusionSpec
    from everyngram.memory import Memory

    class SpyMemory:
        """记录融合验证目标调用 lookup 时传入的上下文。"""

        def __init__(self, inner):
            self.inner = inner
            self.lookup_contexts = []

        def draft(self, ctx, gamma):
            return tuple(self.inner.draft(list(ctx), gamma))

        def lookup(self, ctx):
            self.lookup_contexts.append(tuple(int(t) for t in ctx))
            return self.inner.lookup(list(ctx))

    cycle = [i % 16 for i in range(64)]
    inner = Memory.build(
        cycle, fingerprint="fp", orders=(1, 2, 3), budget=100, continuation_len=8
    )
    spy = SpyMemory(inner)

    greedy_generate(
        FakeCachedModel(),
        [0],
        max_new_tokens=1,
        memory=spy,
        gamma=3,
        use_cache=True,
        fusion=FusionSpec(lam=0.5, memory=spy),
    )

    # 首轮草稿 [1,2,3](上下文 [0])构建 4 个验证分布,查表上下文必须是
    # [0]、[0,1]、[0,1,2]、[0,1,2,3];
    # 旧实现的循环索引差一,实际是 [0]、[0]、[0,1]、[0,1,2]。
    assert spy.lookup_contexts == [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]


def test_cached_speculative_rolls_back_on_rejection():
    from everyngram.memory import Memory

    memory = Memory.build(
        [0, 9] * 4, fingerprint="fp", orders=(1,), budget=100, continuation_len=4
    )
    model = FakeCachedModel()

    out, stats = greedy_generate(
        model, [0], max_new_tokens=4, memory=memory, gamma=2, use_cache=True
    )

    assert out == [1, 2, 3, 4]
    assert stats.accepted == 0
    assert stats.drafted == 2
    # 被拒绝的草稿(9 与 0)已经滚掉,缓存里只剩 [0, 1, 2, 3]
    assert model.last_cache == [0, 1, 2, 3]
    # 每个 token 最多前向一次:1(prefill)+2(被拒草稿)+4(已产出)-1(最后一个没前向)
    assert (
        model.forwarded_tokens
        == 1 + (stats.drafted - stats.accepted) + stats.produced - 1
    )


def test_cached_speculative_with_miss_matches_stateless():
    from everyngram.memory import Memory

    memory = Memory.build([7] * 50, fingerprint="fp", orders=(1,), budget=100)

    plain, _ = greedy_generate(CountingModel(), [0], max_new_tokens=6)
    cached, stats = greedy_generate(
        FakeCachedModel(), [0], max_new_tokens=6, memory=memory, gamma=4, use_cache=True
    )

    assert cached == plain
    assert stats.hits == 0
    assert stats.accept_length == 1.0


def _agreeing_memory():
    from everyngram.memory import Memory

    # 键 (1,) 续写 2,键 (2,) 续写 3:与 +1 模型完全一致
    return Memory.build(
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        fingerprint="fp",
        orders=(1,),
        budget=100,
    )


def _disagreeing_memory():
    from everyngram.memory import Memory

    # 键 (5,) 续写 9:与 +1 模型不一致
    return Memory.build(
        [5, 9, 5, 9, 5, 9, 5, 9],
        fingerprint="fp",
        orders=(1,),
        budget=100,
    )


def test_adaptive_switches_when_probe_agrees_and_stays_lossless():
    from everyngram.decode import adaptive_greedy_generate

    model = FakeCachedModel()
    plain, _ = greedy_generate(model, [1], max_new_tokens=8, use_cache=True)
    out, stats = adaptive_greedy_generate(
        FakeCachedModel(),
        [1],
        max_new_tokens=8,
        memory=_agreeing_memory(),
        gamma=4,
        probe_steps=2,
        min_agree=0.5,
    )

    assert out == plain
    assert stats.switched is True
    assert stats.probe_steps == 2
    assert stats.probe_agreed == 2
    assert stats.produced == 8
    assert stats.as_dict()["switched"] is True


def test_adaptive_stays_plain_when_probe_disagrees():
    from everyngram.decode import adaptive_greedy_generate

    model = FakeCachedModel()
    plain, _ = greedy_generate(model, [5], max_new_tokens=6, use_cache=True)
    out, stats = adaptive_greedy_generate(
        FakeCachedModel(),
        [5],
        max_new_tokens=6,
        memory=_disagreeing_memory(),
        gamma=4,
        probe_steps=2,
        min_agree=0.5,
    )

    assert out == plain
    assert stats.switched is False
    assert stats.probe_steps == 2
    assert stats.probe_agreed == 0


def test_adaptive_probe_longer_than_budget_never_switches():
    from everyngram.decode import adaptive_greedy_generate

    model = FakeCachedModel()
    plain, _ = greedy_generate(model, [1], max_new_tokens=3, use_cache=True)
    out, stats = adaptive_greedy_generate(
        FakeCachedModel(),
        [1],
        max_new_tokens=3,
        memory=_agreeing_memory(),
        gamma=4,
        probe_steps=10,
        min_agree=0.5,
    )

    assert out == plain
    assert stats.switched is False
    assert stats.probe_steps == 3
    assert stats.produced == 3


def test_adaptive_validates_inputs():
    import pytest

    from everyngram.decode import adaptive_greedy_generate

    with pytest.raises(ValueError):
        adaptive_greedy_generate(
            FakeCachedModel(), [1], max_new_tokens=2, memory=None, gamma=4
        )
    with pytest.raises(ValueError):
        adaptive_greedy_generate(
            FakeCachedModel(),
            [1],
            max_new_tokens=2,
            memory=_agreeing_memory(),
            gamma=4,
            min_agree=1.5,
        )
