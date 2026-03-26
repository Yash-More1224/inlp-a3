from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.common.data import MASK, PairDataset, TripleDataset, build_vocab, read_plain_text, split_indices
from src.common.io_utils import ensure_output_dirs, write_text
from src.common.metrics import perplexity_from_loss
from src.common.models import CustomBiLSTM, SimpleSSM
from src.common.seed import set_seed
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb, push_to_hub


def _build_word_chunks(word_ids: list[int], seq_len: int, step: int | None = None) -> list[list[int]]:
    step = step or seq_len
    chunks: list[list[int]] = []
    for start in range(0, max(1, len(word_ids) - seq_len), step):
        end = start + seq_len
        if end >= len(word_ids):
            break
        chunks.append(word_ids[start:end])
    return chunks


def _prepare_task2_data(config: dict):
    plain = read_plain_text(config["data"]["data_dir"])
    # Convert null characters (space placeholder) back to actual spaces for word splitting
    plain = plain.replace('\x00', ' ')
    words = plain.split()
    vocab = build_vocab(words, add_mask=True)
    word_ids = vocab.encode(words)

    seq_len = int(config["data"]["seq_len"])
    chunks = _build_word_chunks(word_ids, seq_len=seq_len, step=int(config["data"].get("step", seq_len)))

    train_idx, val_idx, test_idx = split_indices(len(chunks), config["data"]["train_ratio"], config["data"]["val_ratio"])

    def _pick(indices):
        return [chunks[i] for i in indices]

    train_chunks = _pick(train_idx)
    val_chunks = _pick(val_idx)
    test_chunks = _pick(test_idx)
    return train_chunks, val_chunks, test_chunks, vocab


def _make_mlm_dataset(chunks: list[list[int]], vocab, mask_prob: float):
    mask_id = vocab.stoi[MASK]
    inputs, labels, masks = [], [], []

    for chunk in chunks:
        x = list(chunk)
        y = [-100] * len(chunk)
        z = [0] * len(chunk)
        for i in range(len(chunk)):
            if random.random() < mask_prob:
                y[i] = chunk[i]
                x[i] = mask_id
                z[i] = 1
        if sum(z) == 0:
            j = random.randrange(len(chunk))
            y[j] = chunk[j]
            x[j] = mask_id
            z[j] = 1
        inputs.append(x)
        labels.append(y)
        masks.append(z)

    return TripleDataset(inputs, labels, masks)


def _make_nwp_dataset(chunks: list[list[int]]):
    xs, ys = [], []
    for chunk in chunks:
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return PairDataset(xs, ys)


def _run_bilstm_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def _run_ssm_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def run_task2(config_path: str, mode: str, model_type: str) -> None:
    config = load_config(config_path)
    set_seed(int(config["training"]["seed"]))
    output_dirs = ensure_output_dirs(config["output"]["base_dir"])

    device = config["training"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_chunks, val_chunks, test_chunks, vocab = _prepare_task2_data(config)

    if model_type == "bilstm":
        train_ds = _make_mlm_dataset(train_chunks, vocab, mask_prob=float(config["data"].get("mask_prob", 0.15)))
        val_ds = _make_mlm_dataset(val_chunks, vocab, mask_prob=float(config["data"].get("mask_prob", 0.15)))
        test_ds = _make_mlm_dataset(test_chunks, vocab, mask_prob=float(config["data"].get("mask_prob", 0.15)))
        model = CustomBiLSTM(
            vocab_size=len(vocab.itos),
            embedding_dim=int(config["model"]["embedding_dim"]),
            hidden_size=int(config["model"]["hidden_size"]),
            dropout=float(config["model"]["dropout"]),
        ).to(device)
        run_epoch = _run_bilstm_epoch
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        train_ds = _make_nwp_dataset(train_chunks)
        val_ds = _make_nwp_dataset(val_chunks)
        test_ds = _make_nwp_dataset(test_chunks)
        model = SimpleSSM(
            vocab_size=len(vocab.itos),
            embedding_dim=int(config["model"]["embedding_dim"]),
            state_size=int(config["model"]["state_size"]),
            dropout=float(config["model"]["dropout"]),
        ).to(device)
        run_epoch = _run_ssm_epoch
        criterion = nn.CrossEntropyLoss()

    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))

    ckpt_path = config["output"]["checkpoint_path"]
    ckpt_path = ckpt_path.format(model=model_type)

    use_wandb = bool(config["logging"].get("use_wandb", False))
    if use_wandb:
        init_wandb(project=config["logging"]["project"], config=config, name=f"task2_{model_type}")

    if mode in {"train", "both"}:
        best_val = float("inf")
        best_epoch = -1
        epochs = int(config["training"]["epochs"])

        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

            if use_wandb:
                log_wandb({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch}, step=epoch)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)

        summary_path = Path(output_dirs["logs"]) / f"task2_{model_type}_train_summary.txt"
        write_text(str(summary_path), f"best_epoch={best_epoch}\nbest_val_loss={best_val:.6f}\n")

        if bool(config["hf"].get("push", False)):
            push_to_hub(
                path=ckpt_path,
                repo_id=config["hf"]["repo_id"],
                path_in_repo=f"task2_{model_type}.pt",
                token=config["hf"].get("token"),
            )

    if mode in {"evaluate", "both"}:
        load_checkpoint(ckpt_path, model, optimizer=None, device=device)
        test_loss = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
        ppl = perplexity_from_loss(test_loss)

        if use_wandb:
            log_wandb({"test_loss": test_loss, "perplexity": ppl})
            finish_wandb()

        result_path = Path(output_dirs["results"]) / f"task2_{model_type}.txt"
        write_text(
            str(result_path),
            "\n".join([
                f"model=task2_{model_type}",
                f"test_loss={test_loss:.6f}",
                f"perplexity={ppl:.6f}",
            ]),
        )
