from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
MASK = "<mask>"


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    def encode(self, items: list[str], unk_token: str = UNK) -> list[int]:
        unk_id = self.stoi.get(unk_token, 0)
        return [self.stoi.get(item, unk_id) for item in items]

    def decode(self, ids: list[int], skip_special: bool = False) -> list[str]:
        out: list[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if skip_special and token.startswith("<") and token.endswith(">"):
                continue
            out.append(token)
        return out


def build_vocab(items: list[str], add_mask: bool = False) -> Vocab:
    specials = [PAD, BOS, EOS, UNK]
    if add_mask:
        specials.append(MASK)

    uniq = sorted(set(items))
    itos = specials + [token for token in uniq if token not in specials]
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def read_plain_text(data_dir: str = "data") -> str:
    return (Path(data_dir) / "plain.txt").read_text(encoding="utf-8")


def read_cipher_tokens(filename: str = "cipher_00.txt", data_dir: str = "data") -> list[str]:
    from tqdm import tqdm
    text = (Path(data_dir) / filename).read_text(encoding="utf-8").strip()
    text = text.replace('\n', '').replace(' ', '')
    if not text:
        return []
    return [text[i:i+2] for i in tqdm(range(0, len(text), 2), desc="Chunking Cipher Tokens")]


def chunk_pairs(x: list[int], y: list[int], seq_len: int, step: int | None = None) -> tuple[list[list[int]], list[list[int]]]:
    step = step or seq_len
    x_chunks: list[list[int]] = []
    y_chunks: list[list[int]] = []
    n = min(len(x), len(y))

    for start in range(0, max(1, n - seq_len + 1), step):
        end = start + seq_len
        if end > n:
            break
        x_chunks.append(x[start:end])
        y_chunks.append(y[start:end])

    return x_chunks, y_chunks


def split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[list[int], list[int], list[int]]:
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    idx = list(range(n))
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


class PairDataset(Dataset):
    def __init__(self, xs: list[list[int]], ys: list[list[int]]):
        self.xs = [torch.tensor(x, dtype=torch.long) for x in xs]
        self.ys = [torch.tensor(y, dtype=torch.long) for y in ys]

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


class TripleDataset(Dataset):
    def __init__(self, xs: list[list[int]], ys: list[list[int]], zs: list[list[int]]):
        self.xs = [torch.tensor(x, dtype=torch.long) for x in xs]
        self.ys = [torch.tensor(y, dtype=torch.long) for y in ys]
        self.zs = [torch.tensor(z, dtype=torch.long) for z in zs]

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx], self.zs[idx]
