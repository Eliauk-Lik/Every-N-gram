"""transformers 边界:分词、指纹与模型适配。唯一依赖重库的文件。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def tokenizer_fingerprint(tokenizer) -> str:
    """对词表内容取哈希,用于校验记忆表与分词器是否配套。"""
    vocab = tokenizer.get_vocab()
    digest = hashlib.sha256()
    for token, index in sorted(vocab.items(), key=lambda pair: pair[1]):
        digest.update(f"{index}:{token}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


def load_tokenizer(model_name: str):
    """加载分词器。国内网络必须先设 HF_ENDPOINT=https://hf-mirror.com。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def corpus_token_ids(tokenizer, path: str | Path) -> list[int]:
    """读取语料并分词。目录则按文件名排序读取其中的 .txt。"""
    target = Path(path)
    files = sorted(target.rglob("*.txt")) if target.is_dir() else [target]
    ids: list[int] = []
    for file in files:
        text = file.read_text(encoding="utf-8", errors="ignore")
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
    return ids


class HFModel:
    """HuggingFace 因果语言模型适配器,实现 TokenModel 协议。

    注意:本实现每次都对整段序列重算,复杂度是 O(n^2),
    只用于增量 1 的正确性与接受长度读数,不用于延迟结论。
    """

    def __init__(
        self, model_name: str, device: str | None = None, dtype: str = "float32"
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {}
        if dtype != "auto":
            kwargs["dtype"] = getattr(torch, dtype)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            .to(self.device)
            .eval()
        )

    def next_logits(self, token_ids: Sequence[int]) -> np.ndarray:
        torch = self.torch
        ids = torch.tensor(
            [list(map(int, token_ids))], dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            outputs = self.model(input_ids=ids)
        return outputs.logits[0].float().cpu().numpy()

    def forward(self, token_ids: Sequence[int], cache=None):
        """带缓存前向:只喂新 token,返回新位置 logits 与更新后的缓存。"""
        torch = self.torch
        ids = torch.tensor(
            [list(map(int, token_ids))], dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids, past_key_values=cache, use_cache=True
            )
        logits = outputs.logits[0].float().cpu().numpy()
        if logits.shape[0] != ids.shape[1]:
            # 某些版本在传入缓存时仍返回全量 logits,按约定只保留新位置
            logits = logits[-ids.shape[1] :]
        return logits, outputs.past_key_values

    def crop(self, cache, keep: int):
        """把缓存裁剪到前 keep 个位置。

        DynamicCache 优先用负数 crop(transformers ≥5.18 移除正数语义),
        拿不到长度时才退回旧的正数调用;legacy 的 [(k, v), ...] 结构无法
        就地赋值,按切片重建一层新列表,原对象保持不变;其它类型直接报错。
        越界(keep < 0 或 keep > 当前长度)直接 ValueError,不静默截断。
        """
        keep = int(keep)
        if hasattr(cache, "crop"):
            length = None
            if hasattr(cache, "get_seq_length"):
                try:
                    length = int(cache.get_seq_length())
                except Exception:  # noqa: BLE001 - 拿不到长度就退回旧语义
                    length = None
            if length is not None:
                if keep < 0 or keep > length:
                    raise ValueError(f"keep 必须在 [0, {length}] 内: {keep}")
                remove = length - keep
                if remove:
                    cache.crop(-remove)
                return cache
            if keep < 0:
                raise ValueError(f"keep 必须 >= 0: {keep}")
            cache.crop(keep)
            return cache
        if isinstance(cache, (list, tuple)):
            return [
                (layer[0][:, :, :keep], layer[1][:, :, :keep]) for layer in cache
            ]
        raise TypeError(f"不支持的缓存类型: {type(cache)!r}")
