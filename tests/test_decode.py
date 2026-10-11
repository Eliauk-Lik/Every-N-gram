import numpy as np
import pytest

from everyngram.decode import DecodeStats, greedy_generate


class CountingModel:
    """下一个 token 恒为 (当前 token + 1) % vocab 的确定性模型。"""

    def __init__(self, vocab: int = 16) -> None:
        self.vocab = vocab
        self.calls = 0

    def next_logits(self, token_ids):
        self.calls += 1
        out = np.zeros((len(token_ids), self.vocab), dtype=np.float64)
        for i, token in enumerate(token_ids):
            out[i, (int(token) + 1) % self.vocab] = 1.0
        return out


def test_plain_greedy_counts_up():
    model = CountingModel()
    out, stats = greedy_generate(model, [0], max_new_tokens=5)

    assert out == [1, 2, 3, 4, 5]
    assert stats.steps == 5
    assert stats.produced == 5
    assert stats.accept_length == 1.0
    assert model.calls == 5


def test_plain_greedy_requires_prompt():
    with pytest.raises(ValueError):
        greedy_generate(CountingModel(), [], max_new_tokens=3)


def test_package_exports_public_api():
    import everyngram

    assert hasattr(everyngram, "Memory")
    assert hasattr(everyngram, "greedy_generate")


def test_stats_as_dict_has_all_readings():
    stats = DecodeStats(steps=2, produced=5, drafted=4, accepted=3, lookups=2, hits=2)
    payload = stats.as_dict()

    assert payload["accept_length"] == 2.5
    assert payload["hit_rate"] == 1.0
    assert payload["accept_rate"] == 0.75


from everyngram.memory import Memory

CYCLE = [i % 16 for i in range(200)]


class RunningSumModel:
    """下一个 token 等于整段前缀之和加一,对上下文长度敏感。

    这个模型专门用来抓住"只把最后一个 token 送进模型"的实现错误:
    只看最后一个 token 的假模型漏掉了这个 bug。
    """

    def __init__(self, vocab: int = 32) -> None:
        self.vocab = vocab
        self.calls = 0

    def next_logits(self, token_ids):
        self.calls += 1
        out = np.zeros((len(token_ids), self.vocab), dtype=np.float64)
        total = 0
        for i, token in enumerate(token_ids):
            total += int(token)
            out[i, (total + 1) % self.vocab] = 1.0
        return out


def test_speculative_matches_plain_greedy_with_exact_memory():
    memory = Memory.build(
        CYCLE, fingerprint="fp", orders=(1, 2, 3), budget=1000, continuation_len=16
    )

    plain, _ = greedy_generate(CountingModel(), [0], max_new_tokens=12)

    model = CountingModel()
    spec, stats = greedy_generate(
        model, [0], max_new_tokens=12, memory=memory, gamma=4
    )

    assert spec == plain
    assert stats.lookups == 3
    assert stats.hit_rate == 1.0
    assert stats.accepted == 10
    assert stats.drafted == 12
    assert stats.accept_length == 4.0


def test_speculative_falls_back_on_miss():
    memory = Memory.build([7] * 50, fingerprint="fp", orders=(1,), budget=100)

    plain, _ = greedy_generate(CountingModel(), [0], max_new_tokens=6)

    model = CountingModel()
    spec, stats = greedy_generate(
        model, [0], max_new_tokens=6, memory=memory, gamma=4
    )

    assert spec == plain
    assert stats.hits == 0
    assert stats.accept_length == 1.0
    assert model.calls == 6


def test_speculative_rejects_wrong_drafts_but_stays_exact():
    memory = Memory.build(
        [0, 9] * 4, fingerprint="fp", orders=(1,), budget=100, continuation_len=4
    )
    model = CountingModel()

    spec, stats = greedy_generate(
        model, [0], max_new_tokens=4, memory=memory, gamma=2
    )

    assert spec == [1, 2, 3, 4]
    assert stats.hits == 1
    assert stats.accepted == 0
    assert stats.drafted == 2
    assert model.calls == 4


def test_speculative_stays_exact_with_context_sensitive_model():
    """回归测试:模型对完整上下文敏感时,无损性仍然成立。"""
    plain, _ = greedy_generate(RunningSumModel(), [1, 2, 3], max_new_tokens=10)

    memory = Memory.build(
        plain * 2, fingerprint="fp", orders=(1, 2, 3), budget=1000, continuation_len=16
    )
    spec, stats = greedy_generate(
        RunningSumModel(), [1, 2, 3], max_new_tokens=10, memory=memory, gamma=4
    )

    assert spec == plain
    assert stats.hits > 0
