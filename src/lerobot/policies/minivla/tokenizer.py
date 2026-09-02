import torch


class SimpleTokenizer:
    def __init__(self, vocab=None):
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