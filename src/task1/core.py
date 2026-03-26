from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.config import load_config
from src.common.data import PairDataset, build_vocab, chunk_pairs, read_cipher_tokens, read_plain_text, split_indices
from src.common.io_utils import ensure_output_dirs, write_text
from src.common.metrics import character_accuracy, levenshtein_distance, word_accuracy
from src.common.models import DecryptionModel
from src.common.seed import set_seed
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb, push_to_hub


def _prepare_data(config: dict):
    data_dir = config["data"]["data_dir"]
    plain_text = read_plain_text(data_dir)
    cipher_tokens = read_cipher_tokens("cipher_00.txt", data_dir)

    chars = list(plain_text)
    n = min(len(chars), len(cipher_tokens))
    chars = chars[:n]
    cipher_tokens = cipher_tokens[:n]

    cipher_vocab = build_vocab(cipher_tokens)
    char_vocab = build_vocab(chars)

    x_ids = cipher_vocab.encode(cipher_tokens)
    y_ids = char_vocab.encode(chars)

    seq_len = config["data"]["seq_len"]
    step = config["data"].get("step", seq_len)
    x_chunks, y_chunks = chunk_pairs(x_ids, y_ids, seq_len=seq_len, step=step)

    train_idx, val_idx, test_idx = split_indices(len(x_chunks), config["data"]["train_ratio"], config["data"]["val_ratio"])

    def slice_by_indices(arr, idx):
        return [arr[i] for i in idx]

    train_ds = PairDataset(slice_by_indices(x_chunks, train_idx), slice_by_indices(y_chunks, train_idx))
    val_ds = PairDataset(slice_by_indices(x_chunks, val_idx), slice_by_indices(y_chunks, val_idx))
    test_ds = PairDataset(slice_by_indices(x_chunks, test_idx), slice_by_indices(y_chunks, test_idx))

    return train_ds, val_ds, test_ds, cipher_vocab, char_vocab, plain_text


def _run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Training" if is_train else "Evaluating", leave=False):
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


@torch.no_grad()
def _decode_text(model, cipher_tokens: list[str], cipher_vocab, char_vocab, seq_len: int, device: str):
    model.eval()
    ids = cipher_vocab.encode(cipher_tokens)
    out_chars: list[str] = []

    for start in tqdm(range(0, len(ids), seq_len), desc="Decoding Text"):
        chunk = ids[start : start + seq_len]
        if not chunk:
            continue
        x = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)
        pred = logits.argmax(dim=-1).squeeze(0).tolist()
        out_chars.extend(char_vocab.decode(pred, skip_special=False))

    # Convert null characters (space placeholder) back to spaces
    result = "".join(out_chars)
    result = result.replace('\x00', ' ')
    return result


def run_task1(config_path: str, mode: str, cell_type: str) -> None:
    config = load_config(config_path)
    set_seed(int(config["training"]["seed"]))
    output_dirs = ensure_output_dirs(config["output"]["base_dir"])

    device = config["training"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_ds, val_ds, test_ds, cipher_vocab, char_vocab, plain_text = _prepare_data(config)

    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = DecryptionModel(
        input_vocab_size=len(cipher_vocab.itos),
        output_vocab_size=len(char_vocab.itos),
        embedding_dim=int(config["model"]["embedding_dim"]),
        hidden_size=int(config["model"]["hidden_size"]),
        dropout=float(config["model"]["dropout"]),
        cell_type=cell_type,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    ckpt_path = config["output"]["checkpoint_path"]
    ckpt_path = ckpt_path.format(model=cell_type)

    use_wandb = bool(config["logging"].get("use_wandb", False))
    if use_wandb:
        init_wandb(
            project=config["logging"]["project"],
            config=config,
            name=f"task1_{cell_type}",
        )

    if mode in {"train", "both"}:
        best_val = float("inf")
        best_epoch = -1
        epochs = int(config["training"]["epochs"])

        print(f"\n{'='*60}")
        print(f"Starting Training on {cell_type.upper()} - Device: {device}")
        print(f"Total Epochs: {epochs} | Batch Size: {batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = _run_epoch(model, val_loader, criterion, optimizer=None, device=device)

            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}", end="")

            if use_wandb:
                log_wandb({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch}, step=epoch)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
                print(" ✓ (checkpoint saved)")
            else:
                print()

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Epoch: {best_epoch} | Best Val Loss: {best_val:.6f}")
        print(f"Model saved at: {ckpt_path}")
        print(f"{'='*60}\n")

        summary_path = Path(output_dirs["logs"]) / f"task1_{cell_type}_train_summary.txt"
        write_text(str(summary_path), f"best_epoch={best_epoch}\nbest_val_loss={best_val:.6f}\n")

        if bool(config["hf"].get("push", False)):
            push_to_hub(
                path=ckpt_path,
                repo_id=config["hf"]["repo_id"],
                path_in_repo=f"task1_{cell_type}.pt",
                token=config["hf"].get("token"),
            )

    if mode in {"evaluate", "both"}:
        print(f"\n{'='*60}")
        print(f"Starting Evaluation on {cell_type.upper()}")
        print(f"{'='*60}\n")

        print("Loading checkpoint...")
        load_checkpoint(ckpt_path, model, optimizer=None, device=device)
        
        print("Computing test loss...")
        test_loss = _run_epoch(model, test_loader, criterion, optimizer=None, device=device)

        print("Reading cipher tokens...")
        cipher_tokens = read_cipher_tokens("cipher_00.txt", config["data"]["data_dir"])
        
        print("Decoding text...")
        pred_text = _decode_text(model, cipher_tokens, cipher_vocab, char_vocab, int(config["data"]["seq_len"]), device)

        print("Computing metrics...")
        # Convert null characters back to spaces for metrics comparison
        target_text = plain_text[: len(pred_text)].replace('\x00', ' ')
        metrics = {
            "test_loss": test_loss,
            "char_accuracy": character_accuracy(pred_text, target_text),
            "word_accuracy": word_accuracy(pred_text, target_text),
            "levenshtein": float(levenshtein_distance(pred_text, target_text)),
        }

        if use_wandb:
            log_wandb(metrics)
            finish_wandb()

        result_path = Path(output_dirs["results"]) / f"task1_{cell_type}.txt"
        report = [
            f"model=task1_{cell_type}",
            f"test_loss={metrics['test_loss']:.6f}",
            f"char_accuracy={metrics['char_accuracy']:.6f}",
            f"word_accuracy={metrics['word_accuracy']:.6f}",
            f"levenshtein={metrics['levenshtein']:.0f}",
            "",
            pred_text,
        ]
        write_text(str(result_path), "\n".join(report))

        print(f"\n{'='*60}")
        print(f"Evaluation Complete!")
        print(f"{'='*60}")
        print(f"Test Loss: {metrics['test_loss']:.6f}")
        print(f"Character Accuracy: {metrics['char_accuracy']:.4%}")
        print(f"Word Accuracy: {metrics['word_accuracy']:.4%}")
        print(f"Levenshtein Distance: {metrics['levenshtein']:.0f}")
        print(f"{'='*60}")
        print(f"Results saved to: {result_path}\n")
