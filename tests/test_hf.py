import os

import pytest

from everyngram.hf import corpus_token_ids, tokenizer_fingerprint


class FakeTokenizer:
    """按字符查表的假分词器,用来在无网络环境下测试边界代码。"""

    def __init__(self, vocab):
        self._vocab = dict(vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        return [self._vocab.get(ch, 0) for ch in text]


def test_fingerprint_ignores_insertion_order():
    a = FakeTokenizer({"a": 1, "b": 2})
    b = FakeTokenizer({"b": 2, "a": 1})

    assert tokenizer_fingerprint(a) == tokenizer_fingerprint(b)


def test_fingerprint_changes_with_vocab():
    a = FakeTokenizer({"a": 1, "b": 2})
    c = FakeTokenizer({"a": 1, "c": 2})

    assert tokenizer_fingerprint(a) != tokenizer_fingerprint(c)


def test_corpus_token_ids_reads_txt_in_sorted_order(tmp_path):
    (tmp_path / "a.txt").write_text("ab", encoding="utf-8")
    (tmp_path / "b.txt").write_text("ba", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("zz", encoding="utf-8")

    ids = corpus_token_ids(FakeTokenizer({"a": 1, "b": 2}), tmp_path)

    assert ids == [1, 2, 2, 1]


def test_crop_uses_native_crop_when_available():
    from everyngram.hf import HFModel

    class FakeCache:
        def __init__(self):
            self.kept = None

        def crop(self, keep):
            self.kept = keep
            return self

    cache = FakeCache()

    out = HFModel.crop(None, cache, 3)

    assert out is cache
    assert cache.kept == 3


def test_crop_uses_negative_semantics_when_length_known():
    from everyngram.hf import HFModel

    class NegativeOnlyCache:
        """只接受负数 crop(transformers 5.18 起正数被移除)。"""

        def __init__(self, length):
            self.length = length
            self.removed = 0

        def get_seq_length(self):
            return self.length - self.removed

        def crop(self, keep):
            if keep >= 0:
                raise AssertionError("positive crop is deprecated")
            self.removed += -keep
            return self

    cache = NegativeOnlyCache(10)

    out = HFModel.crop(None, cache, 4)

    assert out is cache
    assert out.get_seq_length() == 4


def test_crop_rejects_out_of_range_keep_when_length_known():
    from everyngram.hf import HFModel

    class Cache:
        def get_seq_length(self):
            return 5

        def crop(self, keep):
            return self

    with pytest.raises(ValueError):
        HFModel.crop(None, Cache(), 8)
    with pytest.raises(ValueError):
        HFModel.crop(None, Cache(), -1)


def test_crop_rebuilds_legacy_tuple_cache_without_mutating_input():
    import numpy as np

    from everyngram.hf import HFModel

    legacy = [(np.zeros((1, 2, 5, 3)), np.zeros((1, 2, 5, 3)))]

    cropped = HFModel.crop(None, legacy, 2)

    assert isinstance(cropped, list) and len(cropped) == 1
    assert cropped[0][0].shape[-2] == 2
    assert cropped[0][1].shape[-2] == 2
    assert legacy[0][0].shape[-2] == 5      # 原缓存不被就地修改


def test_crop_rejects_unknown_cache_type():
    from everyngram.hf import HFModel

    with pytest.raises(TypeError):
        HFModel.crop(None, object(), 2)


@pytest.mark.skipif(
    os.environ.get("EVERYNGRAM_HF_TEST") != "1",
    reason="需要下载模型,设置 EVERYNGRAM_HF_TEST=1 后才跑",
)
def test_real_model_returns_logits_for_every_position():
    from everyngram.hf import HFModel

    model = HFModel("Qwen/Qwen3-0.6B")
    logits = model.next_logits([1, 2, 3])

    assert logits.shape[0] == 3
    assert logits.shape[1] > 1000
