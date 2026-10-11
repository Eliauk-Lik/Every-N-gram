import numpy as np

from everyngram.decode import (
    _residual,
    _sample_index,
    _speculative_draw,
    sample_generate,
)
from everyngram.fuse import softmax as _softmax


class SharpCachedModel:
    """尖峰 logits(30.0 对目标、0 对其它),softmax 后近似确定性。

    采样通路在这种模型上必须退化为贪婪输出。
    """

    def __init__(self, vocab: int = 16) -> None:
        self.vocab = vocab

    def _logits(self, token_ids):
        out = np.zeros((len(token_ids), self.vocab), dtype=np.float64)
        for i, token in enumerate(token_ids):
            out[i, (int(token) + 1) % self.vocab] = 30.0
        return out

    def forward(self, token_ids, cache=None):
        cache = [] if cache is None else list(cache)
        new = [int(t) for t in token_ids]
        return self._logits(cache + new)[len(cache):], cache + new

    def crop(self, cache, keep: int):
        return list(cache)[:keep]


def test_sample_generate_is_reproducible_with_seed():
    a, _ = sample_generate(SharpCachedModel(), [0], max_new_tokens=8, seed=7)
    b, _ = sample_generate(SharpCachedModel(), [0], max_new_tokens=8, seed=7)
    assert a == b


def test_sample_generate_with_sharp_model_matches_greedy():
    out, stats = sample_generate(SharpCachedModel(), [0], max_new_tokens=5, seed=0)
    assert out == [1, 2, 3, 4, 5]
    assert stats.produced == 5


def test_residual_is_normalized_and_nonnegative():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.1, 0.6, 0.3])

    r = _residual(p, q)
    assert abs(r.sum() - 1.0) < 1e-12
    assert (r >= 0).all()
    assert r[1] == 0.0  # q > p 的坐标在残差里必须是 0


def test_softmax_with_low_temperature_concentrates():
    logits = np.array([0.0, 1.0, 2.0])
    sharp = _softmax(logits, temperature=0.1)
    assert int(np.argmax(sharp)) == 2
    assert sharp[2] > 0.99


def test_measured_accept_rate_equals_one_minus_tv():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.1, 0.6, 0.3])
    tv = 0.5 * np.abs(p - q).sum()
    rng = np.random.default_rng(0)

    n = 100_000
    accepted = 0
    for _ in range(n):
        token = _sample_index(rng, q)
        _, ok = _speculative_draw(p, q, token, rng)
        accepted += int(ok)

    assert abs(accepted / n - (1 - tv)) < 0.01


def test_speculative_output_distribution_equals_target():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.1, 0.6, 0.3])
    rng = np.random.default_rng(1)

    n = 100_000
    counts = np.zeros(3)
    for _ in range(n):
        token = _sample_index(rng, q)
        out_token, _ = _speculative_draw(p, q, token, rng)
        counts[out_token] += 1

    empirical = counts / n
    assert 0.5 * np.abs(empirical - p).sum() < 0.01
