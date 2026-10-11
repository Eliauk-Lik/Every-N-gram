import numpy as np

from everyngram.decode import greedy_generate
from everyngram.fuse import FusionSpec
from everyngram.memory import Memory


class CountingModel:
    """下一个 token 恒为 (当前 token + 1) % 16。"""

    def __init__(self) -> None:
        self.vocab = 16

    def next_logits(self, token_ids):
        out = np.zeros((len(token_ids), self.vocab), dtype=np.float64)
        for i, token in enumerate(token_ids):
            out[i, (int(token) + 1) % self.vocab] = 1.0
        return out


class FakeCachedModel:
    """把同一个模型套上缓存接口。"""

    def __init__(self) -> None:
        self.vocab = 16

    def _logits(self, token_ids):
        out = np.zeros((len(token_ids), self.vocab), dtype=np.float64)
        for i, token in enumerate(token_ids):
            out[i, (int(token) + 1) % self.vocab] = 1.0
        return out

    def forward(self, token_ids, cache=None):
        tokens = [] if cache is None else list(cache)
        new = [int(t) for t in token_ids]
        logits = self._logits(tokens + new)[len(tokens):]
        return logits, tokens + new

    def crop(self, cache, keep: int):
        return list(cache)[:keep]


def _memory_with_top_token(top: int) -> Memory:
    """构造一张表:键 (0,) 的最常见续写是 top。"""
    corpus = [0, top] * 20
    return Memory.build(corpus, fingerprint="fp", orders=(1,), budget=100)


def test_fusion_lambda_zero_matches_baseline_stateless():
    memory = _memory_with_top_token(9)
    base, _ = greedy_generate(CountingModel(), [0], max_new_tokens=6)

    fused, _ = greedy_generate(
        CountingModel(), [0], max_new_tokens=6, fusion=FusionSpec(lam=0.0, memory=memory)
    )

    assert fused == base


def test_fusion_lambda_zero_matches_baseline_cached():
    memory = _memory_with_top_token(9)
    base, _ = greedy_generate(FakeCachedModel(), [0], max_new_tokens=6, use_cache=True)

    fused, _ = greedy_generate(
        FakeCachedModel(),
        [0],
        max_new_tokens=6,
        fusion=FusionSpec(lam=0.0, memory=memory),
        use_cache=True,
    )

    assert fused == base


def test_fusion_lambda_one_follows_memory():
    memory = _memory_with_top_token(9)

    fused, _ = greedy_generate(
        CountingModel(), [0], max_new_tokens=3, fusion=FusionSpec(lam=1.0, memory=memory)
    )

    # 表里 (0,) 续写 9、(9,) 续写 0,λ=1 时模型分布被完全覆盖,输出沿表交替
    assert fused == [9, 0, 9]


def test_fusion_with_missing_key_falls_back_to_baseline():
    # 键 (7,) 在 [0] 起步的 +1 链上永远不会出现,四个位置全部未命中
    memory = Memory.build([7] * 20, fingerprint="fp", orders=(1,), budget=100)

    base, _ = greedy_generate(CountingModel(), [0], max_new_tokens=4)
    fused, _ = greedy_generate(
        CountingModel(), [0], max_new_tokens=4, fusion=FusionSpec(lam=0.5, memory=memory)
    )

    assert fused == base


def test_cached_and_stateless_fusion_agree():
    memory = _memory_with_top_token(9)
    spec = FusionSpec(lam=0.5, memory=memory)

    a, _ = greedy_generate(CountingModel(), [0], max_new_tokens=6, fusion=spec)
    b, _ = greedy_generate(
        FakeCachedModel(), [0], max_new_tokens=6, fusion=spec, use_cache=True
    )

    assert a == b


def test_sampling_fusion_lambda_zero_matches_baseline_seed():
    from everyngram.decode import sample_generate

    memory = _memory_with_top_token(9)
    base, _ = sample_generate(FakeCachedModel(), [0], max_new_tokens=6, seed=3)

    fused, _ = sample_generate(
        FakeCachedModel(),
        [0],
        max_new_tokens=6,
        seed=3,
        fusion=FusionSpec(lam=0.0, memory=memory),
    )

    assert fused == base


def test_sampling_fusion_lambda_one_outputs_memory_pattern():
    from everyngram.decode import sample_generate

    memory = Memory.build([0, 9] * 20, fingerprint="fp", orders=(1,), budget=100)

    out, _ = sample_generate(
        FakeCachedModel(),
        [0],
        max_new_tokens=4,
        seed=0,
        fusion=FusionSpec(lam=1.0, memory=memory),
    )

    # λ=1 时靶分布等于表分布,采样结果锁定在 9/0 交替上
    assert out == [9, 0, 9, 0]


def test_sampling_fusion_increases_accept_length():
    from everyngram.decode import sample_generate

    class PlusTwoModel(FakeCachedModel):
        """模型倾向 (当前 token + 2),与表里的 +1 模式不一致。"""

        def _logits(self, token_ids):
            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 2) % 16] = 1.0
            return out

    memory = Memory.build(
        [i % 16 for i in range(64)], fingerprint="fp", orders=(1,), budget=100
    )
    base_stats = sample_generate(
        PlusTwoModel(),
        [0],
        max_new_tokens=8,
        seed=1,
        memory=memory,
        gamma=4,
        fusion=FusionSpec(lam=0.0, memory=memory),
    )[1]
    fused_stats = sample_generate(
        PlusTwoModel(),
        [0],
        max_new_tokens=8,
        seed=1,
        memory=memory,
        gamma=4,
        fusion=FusionSpec(lam=1.0, memory=memory),
    )[1]

    assert fused_stats.accept_length > base_stats.accept_length


def _cycle_memory() -> Memory:
    return Memory.build(
        [i % 16 for i in range(64)], fingerprint="fp", orders=(1,), budget=100
    )


def test_speculative_fusion_lambda_zero_matches_increment2():
    memory = _cycle_memory()

    a, sa = greedy_generate(
        CountingModel(), [0], max_new_tokens=10, memory=memory, gamma=4
    )
    b, sb = greedy_generate(
        FakeCachedModel(),
        [0],
        max_new_tokens=10,
        memory=memory,
        gamma=4,
        fusion=FusionSpec(lam=0.0, memory=memory),
        use_cache=True,
    )

    assert b == a
    assert sb.as_dict() == sa.as_dict()


def test_speculative_fusion_raises_accept_length():
    class PlusTwoModel(FakeCachedModel):
        """靶模型倾向 +2,表里是 +1,λ=0 时草稿必被拒。"""

        def _logits(self, token_ids):
            out = np.zeros((len(token_ids), 16), dtype=np.float64)
            for i, token in enumerate(token_ids):
                out[i, (int(token) + 2) % 16] = 1.0
            return out

    memory = _cycle_memory()
    base = greedy_generate(
        PlusTwoModel(), [0], max_new_tokens=8, memory=memory, gamma=4, use_cache=True
    )[1]
    fused = greedy_generate(
        PlusTwoModel(),
        [0],
        max_new_tokens=8,
        memory=memory,
        gamma=4,
        fusion=FusionSpec(lam=1.0, memory=memory),
        use_cache=True,
    )[1]

    assert base.accept_length == 1.0
    assert fused.accept_length > base.accept_length
