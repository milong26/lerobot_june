import logging
import warnings

import torch

logger = logging.getLogger(__name__)


class VLATokenizerWrapper:
    def __init__(self, tokenizer_path: str = ""):
        self.tokenizer = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, local_files_only=True
                )
                logger.info(f"Loaded VLA tokenizer from {tokenizer_path}")
            except Exception as e:
                warnings.warn(
                    f"Failed to load tokenizer from {tokenizer_path}: {e}. "
                    "Falling back to dummy tokenizer."
                )
                self.tokenizer = None

    def encode(self, texts: str | list[str], max_length: int = 64,
               device: torch.device | None = None) -> dict:
        if self.tokenizer is not None:
            if isinstance(texts, str):
                texts = [texts]
            result = self.tokenizer(
                texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = result["input_ids"]
            attention_mask = result["attention_mask"]
            if device is not None:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            return self._dummy_encode(texts, max_length, device)

    def batch_encode(self, texts: list[str], max_length: int = 64,
                     device: torch.device | None = None) -> dict:
        return self.encode(texts, max_length=max_length, device=device)

    def _dummy_encode(self, texts: str | list[str], max_length: int = 64,
                      device: torch.device | None = None) -> dict:
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
        for i, text in enumerate(texts):
            tokens = str(text).lower().split()[:max_length]
            for j, t in enumerate(tokens):
                input_ids[i, j] = hash(t) % 1000
            attention_mask[i, :len(tokens)] = 1
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class SimpleTokenizer:
    """Legacy tokenizer kept for backward compatibility. Not used by MiniVLAPolicy."""

    def __init__(self, vocab=None):
        warnings.warn(
            "SimpleTokenizer is legacy and not used by MiniVLAPolicy.",
            DeprecationWarning,
        )
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        if vocab is None:
            self.vocab = {self.pad_token: 0, self.unk_token: 1}
        else:
            self.vocab = dict(vocab)
            if self.pad_token not in self.vocab or self.unk_token not in self.vocab:
                raise ValueError("vocab must contain '<pad>' and '<unk>' tokens")
        self.id2token = {v: k for k, v in self.vocab.items()}

    @property
    def pad_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_id(self):
        return self.vocab[self.unk_token]

    @property
    def vocab_size(self):
        return len(self.vocab)

    def build_from_texts(self, texts):
        for text in texts:
            tokens = str(text).lower().split()
            for token in tokens:
                if token not in self.vocab:
                    new_id = len(self.vocab)
                    self.vocab[token] = new_id
                    self.id2token[new_id] = token

    def encode(self, text, max_length=None):
        tokens = str(text).lower().split()
        if not tokens:
            ids = [self.unk_id]
        else:
            ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def batch_encode(self, texts, max_length, device=None):
        all_ids = [self.encode(t, max_length=max_length) for t in texts]
        max_len = max(len(ids) for ids in all_ids)
        if max_length is not None:
            max_len = min(max_len, max_length)
        padded = []
        for ids in all_ids:
            if len(ids) < max_len:
                ids = ids + [self.pad_id] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
            padded.append(ids)
        result = torch.tensor(padded, dtype=torch.long)
        if device is not None:
            result = result.to(device)
        return result