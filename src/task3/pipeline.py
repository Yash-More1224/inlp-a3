from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from src.common.config import load_config
from src.common.data import MASK, build_vocab, read_cipher_tokens, read_plain_text
from src.common.io_utils import ensure_output_dirs, write_text
from src.common.metrics import character_accuracy, corpus_bleu, levenshtein_distance, rouge_l_f1, word_accuracy, chrf_score
from src.common.models import CustomBiLSTM, DecryptionModel, SimpleSSM
from src.common.seed import set_seed
from src.utils.checkpoints import load_checkpoint
from src.utils.hf_wandb import finish_wandb, init_wandb, log_wandb, pull_from_hub


def _resolve_checkpoint(hf_cfg: dict, local_path: str, filename: str) -> str:
    repo_id = hf_cfg.get("repo_id", "")
    token = hf_cfg.get("token")
    if repo_id:
        try:
            print(f"  [HF] Downloading {filename} from {repo_id}")
            downloaded_path = pull_from_hub(repo_id=repo_id, filename=filename, local_dir=str(Path(local_path).parent), token=token)
            print(f"  [HF] Successfully downloaded {filename}")
            return downloaded_path
        except Exception as e:
            print(f"  [HF] Failed to download from hub: {e}")
            print(f"  [HF] Falling back to local path: {local_path}")
            return local_path
    print(f"  [Local] Using local checkpoint: {local_path}")
    return local_path


@torch.no_grad()
def _decrypt_text(
    model, cipher_tokens: list[str], cipher_vocab, char_vocab,
    seq_len: int, device: str, batch_size: int = 256,
    oov_fallback_id: int | None = None,
) -> tuple[str, list[float]]:
    """Decrypt cipher tokens using fully-batched GPU inference.
    
    Args:
        oov_fallback_id: Optional char ID to use for out-of-vocab cipher tokens.
                          If None, uses the most common character (space).
    """
    model.eval()
    
    # Handle OOV cipher tokens by mapping them to <unk> or a known token
    unk_id = cipher_vocab.stoi.get("<unk>", 0)
    valid_ids = set(range(len(cipher_vocab.itos)))
    
    # Encode with OOV handling
    ids = []
    for tok in cipher_tokens:
        idx = cipher_vocab.stoi.get(tok, unk_id)
        if idx not in valid_ids:
            idx = unk_id
        ids.append(idx)
    
    # Slice the full sequence into non-overlapping chunks
    chunks = [ids[s : s + seq_len] for s in range(0, len(ids), seq_len) if ids[s : s + seq_len]]
    if not chunks:
        return "", []

    out_chars: list[str] = []
    confs: list[float] = []

    # Process chunks in large GPU batches
    for b_start in tqdm(
        range(0, len(chunks), batch_size),
        desc="Decryption",
        leave=False,
        dynamic_ncols=False,
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}]"
    ):
        batch_chunks = chunks[b_start : b_start + batch_size]
        # Pad to the longest chunk in this mini-batch
        max_len = max(len(c) for c in batch_chunks)
        pad_id = cipher_vocab.stoi.get("<pad>", 0)
        padded = [c + [pad_id] * (max_len - len(c)) for c in batch_chunks]

        x = torch.tensor(padded, dtype=torch.long, device=device)  # (B, T)
        logits = model(x)                                           # (B, T, vocab)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)   # (B, T)
        maxp  = probs.max(dim=-1).values  # (B, T)

        # Unpad and collect — only keep positions that were real tokens
        for i, orig_chunk in enumerate(batch_chunks):
            real_len = len(orig_chunk)
            out_chars.extend(char_vocab.decode(preds[i, :real_len].tolist(), skip_special=False))
            confs.extend(maxp[i, :real_len].tolist())

    result = "".join(out_chars).replace("\x00", " ")
    return result, confs


def _find_low_conf_char_positions(text: str, char_conf: list[float], threshold: float) -> list[int]:
    """Find character positions with confidence below threshold.
    
    Also includes adjacent positions to create context for better correction.
    """
    chars = list(text)
    if not chars:
        return []

    positions = []
    for idx, conf in enumerate(char_conf):
        if idx < len(chars) and conf < threshold:
            positions.append(idx)
    
    # If very few positions found, also include medium confidence positions
    if len(positions) < len(chars) * 0.05:  # Less than 5% marked
        for idx, conf in enumerate(char_conf):
            if idx < len(chars) and idx not in positions and conf < threshold + 0.2:
                positions.append(idx)
        positions.sort()
    
    return positions


@torch.no_grad()
def _correct_with_bilstm(
    text: str, low_pos: list[int], lm_model, lm_vocab,
    seq_len: int, device: str, batch_size: int = 256,
    aux_models: list | None = None,
) -> str:
    """Correct low-confidence chars with BiLSTM.

    aux_models: optional list of (model, device_str) for additional GPUs.
    Batches are round-robined across [primary] + aux_models.  Each replica
    computes argmax independently — no cross-GPU logits gather, no OOM.
    """
    chars = list(text)
    if not chars:
        return text

    mask_id = lm_vocab.stoi[MASK]
    unk_id = lm_vocab.stoi.get("<unk>", 0)
    total = len(low_pos)

    # Build replica list: primary first, then any extras
    replicas = [(lm_model, device)] + (aux_models or [])
    n_replicas = len(replicas)

    pbar = tqdm(
        total=total,
        desc=f"BiLSTM Correction ({n_replicas} GPU{'s' if n_replicas > 1 else ''})",
        leave=True,
        miniters=1000,
        mininterval=30.0,
        dynamic_ncols=False,
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n}/{total} [{elapsed}<{remaining}]",
    )

    for b_idx, b_start in enumerate(range(0, total, batch_size)):
        # Round-robin across GPU replicas
        curr_model, curr_device = replicas[b_idx % n_replicas]
        batch = low_pos[b_start : b_start + batch_size]

        windows_list: list[list[int]] = []
        local_pos_list: list[int] = []
        pos_list: list[int] = []

        for pos in batch:
            left = max(0, pos - seq_len // 2)
            right = min(len(chars), left + seq_len)
            left = max(0, right - seq_len)

            window = chars[left:right]
            x = [lm_vocab.stoi.get(c, unk_id) for c in window]
            local_pos = pos - left

            if 0 <= local_pos < len(x):
                x[local_pos] = mask_id
                windows_list.append(x)
                local_pos_list.append(local_pos)
                pos_list.append(pos)

        if not windows_list:
            pbar.update(len(batch))
            continue

        max_len = max(len(w) for w in windows_list)
        padded = [w + [mask_id] * (max_len - len(w)) for w in windows_list]

        xt = torch.tensor(padded, dtype=torch.long, device=curr_device)  # (B, T)
        logits = curr_model(xt)                                           # (B, T, vocab)

        # argmax on the same GPU — only scalars cross to CPU, not the full logits
        for i, (local_pos, pos) in enumerate(zip(local_pos_list, pos_list)):
            pred_id = int(logits[i, local_pos].argmax().item())
            pred_char = lm_vocab.itos[pred_id]
            if not pred_char.startswith("<"):
                chars[pos] = pred_char

        pbar.update(len(batch))

    pbar.close()
    return "".join(chars)


@torch.no_grad()
def _correct_with_ssm(
    text: str, low_pos: list[int], lm_model, lm_vocab,
    seq_len: int, device: str, batch_size: int = 256,
    aux_models: list | None = None,
) -> str:
    """Correct low-confidence chars using the SSM language model.

    Uses a capped sliding-window context (last `seq_len` tokens before `pos`)
    so every sequence in the batch has the same bounded length, avoiding the
    O(N) padding blow-up that made early positions tiny and late positions huge.
    """
    chars = list(text)
    if len(chars) < 2:
        return text

    unk_id = lm_vocab.stoi.get("<unk>", 0)
    mask_id = lm_vocab.stoi.get(MASK, 0)
    ids = [lm_vocab.stoi.get(c, unk_id) for c in chars]

    total = len(low_pos)

    # Build replica list: primary first, then any extras
    replicas = [(lm_model, device)] + (aux_models or [])
    n_replicas = len(replicas)

    pbar = tqdm(
        total=total,
        desc=f"SSM Correction ({n_replicas} GPU{'s' if n_replicas > 1 else ''})",
        leave=True,
        miniters=1000,
        mininterval=30.0,
        dynamic_ncols=False,
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n}/{total} [{elapsed}<{remaining}]",
    )

    for b_idx, b_start in enumerate(range(0, total, batch_size)):
        curr_model, curr_device = replicas[b_idx % n_replicas]
        batch = low_pos[b_start : b_start + batch_size]

        contexts: list[list[int]] = []
        valid_pos: list[int] = []

        for pos in batch:
            if pos == 0:
                continue
            # Use only the last seq_len tokens before pos (sliding window)
            start = max(0, pos - seq_len)
            context = ids[start:pos]   # length ≤ seq_len, always > 0
            if context:
                contexts.append(context)
                valid_pos.append(pos)

        if not contexts:
            pbar.update(len(batch))
            continue

        # Pad left so every context is seq_len tokens (left-padding mirrors
        # what a causal model expects when context is shorter)
        padded = [[mask_id] * (seq_len - len(c)) + c for c in contexts]

        x = torch.tensor(padded, dtype=torch.long, device=curr_device)  # (B, seq_len)
        logits = curr_model(x)                                           # (B, seq_len, vocab)

        for i, pos_val in enumerate(valid_pos):
            # Prediction for the next char comes from the last real context position
            last_idx = seq_len - 1  # always the rightmost token after left-padding
            pred_id = int(logits[i, last_idx].argmax().item())
            pred_char = lm_vocab.itos[pred_id]
            if not pred_char.startswith("<"):
                chars[pos_val] = pred_char

        pbar.update(len(batch))

    pbar.close()
    return "".join(chars)


def _compute_metrics(pred: str, target: str) -> dict[str, float]:
    """Compute metrics with proper alignment between prediction and target."""
    if not pred or not target:
        return {
            "char_accuracy": 0.0,
            "word_accuracy": 0.0,
            "levenshtein": float(len(target) if target else 0),
            "bleu": 0.0,
            "rouge_l": 0.0,
            "chrf": 0.0,
        }
    
    # Ensure we're comparing same-length strings
    # The cipher and plain should have 1:1 mapping, but lengths might differ
    # due to noise or truncation. Take the minimum to ensure fair comparison.
    min_len = min(len(pred), len(target))
    pred_aligned = pred[:min_len]
    target_aligned = target[:min_len]
    
    return {
        "char_accuracy": character_accuracy(pred_aligned, target_aligned),
        "word_accuracy": word_accuracy(pred_aligned, target_aligned),
        "levenshtein": float(levenshtein_distance(pred, target)),
        "bleu": corpus_bleu(pred, target),
        "rouge_l": rouge_l_f1(pred, target),
        "chrf": chrf_score(pred, target),
    }


def main(config_path: str, mode: str) -> None:
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    output_dirs = ensure_output_dirs(config["output"]["base_dir"])

    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Print HF repo info prominently
    dec_hf_repo = config["decryption_model"].get("hf", {}).get("repo_id", "")
    lm_hf_repo = config["language_model"].get("hf", {}).get("repo_id", "")
    print(f"\n{'='*60}")
    print(f"HuggingFace Model Configuration")
    print(f"{'='*60}")
    if dec_hf_repo:
        print(f"✓ Decryption model HF repo: {dec_hf_repo}")
    else:
        print(f"✗ Decryption model: Using local path only")
    if lm_hf_repo:
        print(f"✓ Language model HF repo: {lm_hf_repo}")
    else:
        print(f"✗ Language model: Using local path only")
    print(f"{'='*60}\n")

    num_gpus = torch.cuda.device_count() if device == "cuda" else 0
    if num_gpus > 1:
        print(f"✓ Found {num_gpus} GPUs — enabling DataParallel across all of them")

    plain = read_plain_text(config["data"]["data_dir"])
    cipher_clean = read_cipher_tokens("cipher_00.txt", config["data"]["data_dir"], verbose=False)
    
    # Fix: cipher_clean is a list of tokens, so len(cipher_clean) = num tokens
    # Each cipher token maps to 1 character, so the lengths should match 1:1
    # Build vocab from all cipher tokens (full length)
    cipher_vocab = build_vocab(cipher_clean)
    
    # For char_vocab, use the full plain text (cipher_clean has same number of tokens as chars in plain)
    # The cipher was built such that each token maps to one character position
    plain_chars_full = list(plain.replace('\x00', ' '))
    # Truncate to match cipher token count if needed
    if len(plain_chars_full) > len(cipher_clean):
        plain_chars_full = plain_chars_full[:len(cipher_clean)]
    char_vocab = build_vocab(plain_chars_full)

    # Build language model vocabulary like Task 2: char-level with spaces restored
    plain_chars = list(plain.replace('\x00', ' '))
    lm_vocab = build_vocab(plain_chars, add_mask=True)

    dec_model = DecryptionModel(
        input_vocab_size=len(cipher_vocab.itos),
        output_vocab_size=len(char_vocab.itos),
        embedding_dim=int(config["decryption_model"]["embedding_dim"]),
        hidden_size=int(config["decryption_model"]["hidden_size"]),
        dropout=float(config["decryption_model"]["dropout"]),
        cell_type=config["decryption_model"]["cell_type"],
        num_layers=int(config["decryption_model"].get("num_layers", 1)),
    ).to(device)

    lm_type = config["language_model"]["type"]
    if lm_type == "bilstm":
        lm_model = CustomBiLSTM(
            vocab_size=len(lm_vocab.itos),
            embedding_dim=int(config["language_model"]["embedding_dim"]),
            hidden_size=int(config["language_model"]["hidden_size"]),
            dropout=float(config["language_model"]["dropout"]),
        ).to(device)
    else:
        lm_model = SimpleSSM(
            vocab_size=len(lm_vocab.itos),
            embedding_dim=int(config["language_model"]["embedding_dim"]),
            state_size=int(config["language_model"]["state_size"]),
            dropout=float(config["language_model"]["dropout"]),
        ).to(device)

    # Load checkpoints on bare models BEFORE wrapping with DataParallel.
    # DataParallel adds a "module." prefix to all state dict keys, so loading
    # a checkpoint saved without it into a wrapped model causes key mismatches.
    dec_local = config["decryption_model"]["checkpoint_path"]
    dec_file = config["decryption_model"].get("hf_filename", Path(dec_local).name)
    dec_ckpt = _resolve_checkpoint(config["decryption_model"].get("hf", {}), dec_local, dec_file)
    load_checkpoint(dec_ckpt, dec_model, optimizer=None, device=device)

    lm_local = config["language_model"]["checkpoint_path"]
    lm_file = config["language_model"].get("hf_filename", Path(lm_local).name)
    lm_ckpt = _resolve_checkpoint(config["language_model"].get("hf", {}), lm_local, lm_file)
    load_checkpoint(lm_ckpt, lm_model, optimizer=None, device=device)

    # Manual multi-GPU: replicate LM weights to each additional GPU.
    # We avoid DataParallel because it gathers the FULL logits tensor
    # (batch × seq_len × vocab_size) on GPU 0, which OOMs for large vocabs.
    # Instead, each GPU replica processes its own batches independently and
    # only returns argmax indices (scalars) — zero cross-GPU tensor transfer.
    lm_aux: list = []
    if num_gpus > 1:
        import copy
        primary_device = "cuda:0"
        lm_model = lm_model.to(primary_device)
        device = primary_device
        for gpu_idx in range(1, num_gpus):
            alt_dev = f"cuda:{gpu_idx}"
            replica = copy.deepcopy(lm_model).to(alt_dev)
            replica.eval()
            lm_aux.append((replica, alt_dev))
        print(f"✓ LM replicated to {num_gpus} GPUs: batches round-robined across cuda:0..cuda:{num_gpus - 1}")

    noisy_files = config["data"].get("noisy_files", ["cipher_01.txt", "cipher_02.txt", "cipher_03.txt", "cipher_04.txt"])
    seq_len = int(config["data"].get("seq_len", 128))
    conf_threshold = float(config.get("confidence_threshold", 0.6))
    batch_size = int(config.get("batch_size", 256))
    lm_seq_len = int(config["language_model"].get("seq_len", 32))

    sections = [f"pipeline=task3_{lm_type}", f"mode={mode}"]

    # WandB initialization
    use_wandb = bool(config.get("logging", {}).get("use_wandb", False))
    if use_wandb:
        try:
            init_wandb(
                project=config["logging"]["project"],
                config=config,
                name=f"task3_{lm_type}",
            )
            print("✓ WandB initialized successfully")
        except Exception as e:
            print(f"⚠️ WandB initialization failed: {e}")
            print("Continuing without WandB logging...")
            use_wandb = False

    print(f"\n{'='*60}")
    print(f"Starting Task 3 Evaluation with {lm_type.upper()} Language Model")
    print(f"Processing {len(noisy_files)} noisy cipher files")
    print(f"{'='*60}\n")

    for filename in tqdm(noisy_files, desc="Processing Files"):
        data_dir = config["data"]["data_dir"]
        print(f"\n  Processing {filename}...")

        cipher_noisy = read_cipher_tokens(filename, data_dir, verbose=False)

        pred_raw, conf = _decrypt_text(
            dec_model, cipher_noisy, cipher_vocab, char_vocab,
            seq_len=seq_len, device=device, batch_size=batch_size
        )
        
        # Debug: Show decryption output sample
        print(f"    Decrypted {len(pred_raw)} chars, avg conf: {sum(conf)/len(conf):.3f}" if conf else "    No confidence scores")
        
        low_pos = _find_low_conf_char_positions(pred_raw, conf, conf_threshold)
        print(f"    Found {len(low_pos)} low-confidence positions ({100*len(low_pos)/max(len(pred_raw),1):.1f}%)")

        if lm_type == "bilstm":
            pred_corrected = _correct_with_bilstm(
                pred_raw, low_pos, lm_model, lm_vocab,
                seq_len=lm_seq_len, device=device, batch_size=batch_size,
                aux_models=lm_aux if lm_aux else None,
            )
        else:
            pred_corrected = _correct_with_ssm(
                pred_raw, low_pos, lm_model, lm_vocab,
                seq_len=lm_seq_len, device=device, batch_size=batch_size,
                aux_models=lm_aux if lm_aux else None,
            )
        
        # Count how many characters were actually changed
        changes = sum(1 for a, b in zip(pred_raw, pred_corrected) if a != b)
        print(f"    Made {changes} corrections")

        m_raw = _compute_metrics(pred_raw, plain)
        m_corr = _compute_metrics(pred_corrected, plain)

        sections.extend(
            [
                "",
                f"file={filename}",
                f"baseline_char_accuracy={m_raw['char_accuracy']:.6f}",
                f"baseline_word_accuracy={m_raw['word_accuracy']:.6f}",
                f"baseline_levenshtein={m_raw['levenshtein']:.0f}",
                f"baseline_bleu={m_raw['bleu']:.6f}",
                f"baseline_rouge_l={m_raw['rouge_l']:.6f}",
                f"baseline_chrf={m_raw['chrf']:.6f}",
                f"corrected_char_accuracy={m_corr['char_accuracy']:.6f}",
                f"corrected_word_accuracy={m_corr['word_accuracy']:.6f}",
                f"corrected_levenshtein={m_corr['levenshtein']:.0f}",
                f"corrected_bleu={m_corr['bleu']:.6f}",
                f"corrected_rouge_l={m_corr['rouge_l']:.6f}",
                f"corrected_chrf={m_corr['chrf']:.6f}",
            ]
        )

        if use_wandb:
            noise_level = Path(filename).stem  # e.g. "cipher_01"
            log_wandb({
                f"{noise_level}/baseline/char_accuracy": m_raw["char_accuracy"],
                f"{noise_level}/baseline/word_accuracy": m_raw["word_accuracy"],
                f"{noise_level}/baseline/levenshtein": m_raw["levenshtein"],
                f"{noise_level}/baseline/bleu": m_raw["bleu"],
                f"{noise_level}/baseline/rouge_l": m_raw["rouge_l"],
                f"{noise_level}/baseline/chrf": m_raw["chrf"],
                f"{noise_level}/corrected/char_accuracy": m_corr["char_accuracy"],
                f"{noise_level}/corrected/word_accuracy": m_corr["word_accuracy"],
                f"{noise_level}/corrected/levenshtein": m_corr["levenshtein"],
                f"{noise_level}/corrected/bleu": m_corr["bleu"],
                f"{noise_level}/corrected/rouge_l": m_corr["rouge_l"],
                f"{noise_level}/corrected/chrf": m_corr["chrf"],
                f"{noise_level}/improvement/char_accuracy": m_corr["char_accuracy"] - m_raw["char_accuracy"],
                f"{noise_level}/improvement/word_accuracy": m_corr["word_accuracy"] - m_raw["word_accuracy"],
                f"{noise_level}/improvement/bleu": m_corr["bleu"] - m_raw["bleu"],
                f"{noise_level}/improvement/rouge_l": m_corr["rouge_l"] - m_raw["rouge_l"],
                f"{noise_level}/improvement/chrf": m_corr["chrf"] - m_raw["chrf"],
            })

    output_file = config["output"].get("result_file", f"task3_{lm_type}.txt")
    result_path = Path(output_dirs["results"]) / output_file
    print(f"\n✓ Writing results to {result_path}")
    write_text(str(result_path), "\n".join(sections))
    print(f"✓ Task 3 Evaluation Complete!\n")

    if use_wandb:
        finish_wandb()

    input_file = config.get("input_file")
    output_text_file = config.get("output_text_file")
    if input_file and output_text_file:
        cipher_custom = read_cipher_tokens(input_file, config["data"]["data_dir"], verbose=False)
        pred_raw, conf = _decrypt_text(
            dec_model, cipher_custom, cipher_vocab, char_vocab,
            seq_len=seq_len, device=device, batch_size=batch_size
        )
        low_pos = _find_low_conf_char_positions(pred_raw, conf, conf_threshold)
        if lm_type == "bilstm":
            pred_custom = _correct_with_bilstm(
                pred_raw, low_pos, lm_model, lm_vocab,
                seq_len=lm_seq_len, device=device, batch_size=batch_size
            )
        else:
            pred_custom = _correct_with_ssm(
                pred_raw, low_pos, lm_model, lm_vocab,
                seq_len=lm_seq_len, device=device, batch_size=batch_size
            )
        write_text(output_text_file, pred_custom)
