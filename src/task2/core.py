from __future__ import annotations

import pickle
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.config import load_config
from src.common.data import MASK, PairDataset, TripleDataset, build_vocab, read_plain_text, split_indices
from src.common.io_utils import ensure_output_dirs, write_text
from src.common.metrics import perplexity_from_loss
from src.common.models import CustomBiLSTM, SimpleSSM
from src.common.seed import set_seed
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb, push_to_hub


def _get_data_cache_path(output_dirs: dict) -> str:
    """Get the cache file path for processed data."""
    cache_dir = Path(output_dirs["logs"]) / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / "task2_processed_data.pkl")


def _load_data_cache(cache_path: str) -> dict | None:
    """Load processed data from cache if it exists."""
    path = Path(cache_path)
    if path.exists():
        print("✓ Loading processed data from cache...")
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def _save_data_cache(cache_path: str, data: dict) -> None:
    """Save processed data to cache."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Saved processed data to cache: {cache_path}")


def _build_word_chunks(word_ids: list[int], seq_len: int, step: int | None = None) -> list[list[int]]:
    step = step or seq_len
    chunks: list[list[int]] = []
    for start in range(0, max(1, len(word_ids) - seq_len), step):
        end = start + seq_len
        if end >= len(word_ids):
            break
        chunks.append(word_ids[start:end])
    return chunks


def _prepare_task2_data(config: dict, output_dirs: dict):
    cache_path = _get_data_cache_path(output_dirs)
    cached_data = _load_data_cache(cache_path)
    
    if cached_data is not None:
        # Verify the cached data matches current config
        if (cached_data['seq_len'] == int(config["data"]["seq_len"]) and
            cached_data['step'] == int(config["data"].get("step", config["data"]["seq_len"])) and
            cached_data['train_ratio'] == config["data"]["train_ratio"] and
            cached_data['val_ratio'] == config["data"]["val_ratio"]):
            return (cached_data['train_chunks'], cached_data['val_chunks'], 
                   cached_data['test_chunks'], cached_data['vocab'])
    
    # Process data if not cached or cache is invalid
    print("Processing data...")
    plain = read_plain_text(config["data"]["data_dir"])
    # Convert null characters (space placeholder) back to actual spaces for word splitting
    plain = plain.replace('\x00', ' ')
    words = plain.split()
    vocab = build_vocab(words, add_mask=True)
    word_ids = vocab.encode(words)

    seq_len = int(config["data"]["seq_len"])
    step = int(config["data"].get("step", seq_len))
    chunks = _build_word_chunks(word_ids, seq_len=seq_len, step=step)

    train_idx, val_idx, test_idx = split_indices(len(chunks), config["data"]["train_ratio"], config["data"]["val_ratio"])

    def _pick(indices):
        return [chunks[i] for i in indices]

    train_chunks = _pick(train_idx)
    val_chunks = _pick(val_idx)
    test_chunks = _pick(test_idx)
    
    # Cache the processed data
    cache_data = {
        'train_chunks': train_chunks,
        'val_chunks': val_chunks,
        'test_chunks': test_chunks,
        'vocab': vocab,
        'seq_len': seq_len,
        'step': step,
        'train_ratio': config["data"]["train_ratio"],
        'val_ratio': config["data"]["val_ratio"]
    }
    _save_data_cache(cache_path, cache_data)
    
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

    train_chunks, val_chunks, test_chunks, vocab = _prepare_task2_data(config, output_dirs)

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

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0))
    )

    ckpt_path = config["output"]["checkpoint_path"]
    ckpt_path = ckpt_path.format(model=model_type)

    use_wandb = bool(config["logging"].get("use_wandb", False))
    if use_wandb:
        try:
            init_wandb(project=config["logging"]["project"], config=config, name=f"task2_{model_type}")
            print("✓ WandB initialized successfully")
        except Exception as e:
            print(f"⚠️ WandB initialization failed: {e}")
            print("Continuing without WandB logging...")
            use_wandb = False

    if mode in {"train", "both"}:
        best_val = float("inf")
        best_epoch = -1
        epochs = int(config["training"]["epochs"])
        patience = int(config["training"].get("patience", epochs))  # Early stopping patience
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"Starting Training on {model_type.upper()} - Device: {device}")
        print(f"Total Epochs: {epochs} | Batch Size: {batch_size} | Patience: {patience}")
        print(f"{'='*60}\n")

        for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs", unit="epoch"):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}", end="")

            if use_wandb:
                log_wandb({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch}, step=epoch)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                patience_counter = 0  # Reset patience counter
                save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
                print(" ✓ (checkpoint saved)")
            else:
                patience_counter += 1
                print(f" | Patience: {patience_counter}/{patience}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
                    break

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Epoch: {best_epoch} | Best Val Loss: {best_val:.6f}")
        print(f"Model saved at: {ckpt_path}")
        print(f"{'='*60}\n")

        summary_path = Path(output_dirs["logs"]) / f"task2_{model_type}_train_summary.txt"
        write_text(str(summary_path), f"best_epoch={best_epoch}\nbest_val_loss={best_val:.6f}\n")

        if use_wandb:
            finish_wandb()

        if bool(config["hf"].get("push", False)):
            push_to_hub(
                path=ckpt_path,
                repo_id=config["hf"]["repo_id"],
                path_in_repo=f"task2_{model_type}.pt",
                token=config["hf"].get("token"),
            )

    if mode in {"evaluate", "both"}:
        print(f"\n{'='*60}")
        print(f"Starting Evaluation on {model_type.upper()}")
        print(f"{'='*60}\n")

        print("Loading checkpoint...")
        load_checkpoint(ckpt_path, model, optimizer=None, device=device)
        
        print("Computing test loss...")
        test_loss = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
        
        print("Computing perplexity...")
        ppl = perplexity_from_loss(test_loss)

        # Get best_epoch for WandB logging step
        best_epoch = -1
        summary_path = Path(output_dirs["logs"]) / f"task2_{model_type}_train_summary.txt"
        if summary_path.exists():
            try:
                summary_content = summary_path.read_text()
                for line in summary_content.strip().split('\n'):
                    if line.startswith('best_epoch='):
                        best_epoch = int(line.split('=')[1])
                        break
            except Exception:
                pass  # Use default if reading fails

        if use_wandb:
            log_wandb({"test_loss": test_loss, "perplexity": ppl}, step=best_epoch if best_epoch > 0 else 1)
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

        print(f"\n{'='*60}")
        print(f"Evaluation Complete!")
        print(f"{'='*60}")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Perplexity: {ppl:.6f}")
        print(f"{'='*60}")
        print(f"Results saved to: {result_path}\n")
