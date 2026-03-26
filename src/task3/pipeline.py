from __future__ import annotations

from pathlib import Path

import torch

from src.common.config import load_config
from src.common.data import MASK, build_vocab, read_cipher_tokens, read_plain_text
from src.common.io_utils import ensure_output_dirs, write_text
from src.common.metrics import character_accuracy, corpus_bleu, levenshtein_distance, rouge_l_f1, word_accuracy
from src.common.models import CustomBiLSTM, DecryptionModel, SimpleSSM
from src.common.seed import set_seed
from src.utils.checkpoints import load_checkpoint
from src.utils.hf_wandb import pull_from_hub


def _resolve_checkpoint(hf_cfg: dict, local_path: str, filename: str) -> str:
    repo_id = hf_cfg.get("repo_id", "")
    token = hf_cfg.get("token")
    if repo_id:
        try:
            return pull_from_hub(repo_id=repo_id, filename=filename, local_dir=str(Path(local_path).parent), token=token)
        except Exception:
            return local_path
    return local_path


@torch.no_grad()
def _decrypt_text(model, cipher_tokens: list[str], cipher_vocab, char_vocab, seq_len: int, device: str) -> tuple[str, list[float]]:
    model.eval()
    ids = cipher_vocab.encode(cipher_tokens)
    out_chars: list[str] = []
    confs: list[float] = []

    for start in range(0, len(ids), seq_len):
        chunk = ids[start : start + seq_len]
        if not chunk:
            continue
        x = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).squeeze(0).tolist()
        mx = probs.max(dim=-1).values.squeeze(0).tolist()
        out_chars.extend(char_vocab.decode(pred, skip_special=False))
        confs.extend([float(v) for v in mx])

    # Convert null characters (space placeholder) back to spaces
    result = "".join(out_chars)
    result = result.replace('\x00', ' ')
    return result, confs


def _find_low_conf_word_positions(text: str, char_conf: list[float], threshold: float) -> list[int]:
    words = text.split()
    if not words:
        return []

    positions = []
    cursor = 0
    for idx, w in enumerate(words):
        start = text.find(w, cursor)
        end = start + len(w)
        cursor = end
        if start < 0:
            continue
        vals = char_conf[start:end] if end <= len(char_conf) else []
        avg = sum(vals) / max(1, len(vals)) if vals else 0.0
        if avg < threshold:
            positions.append(idx)
    return positions


@torch.no_grad()
def _correct_with_bilstm(text: str, low_pos: list[int], lm_model, lm_vocab, seq_len: int, device: str) -> str:
    words = text.split()
    if not words:
        return text

    mask_id = lm_vocab.stoi[MASK]
    unk_id = lm_vocab.stoi.get("<unk>", 0)

    for pos in low_pos:
        left = max(0, pos - seq_len // 2)
        right = min(len(words), left + seq_len)
        left = max(0, right - seq_len)

        window = words[left:right]
        x = [lm_vocab.stoi.get(w, unk_id) for w in window]
        local_pos = pos - left
        if 0 <= local_pos < len(x):
            x[local_pos] = mask_id
            xt = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
            logits = lm_model(xt)
            pred_id = int(logits[0, local_pos].argmax().item())
            pred_word = lm_vocab.itos[pred_id]
            if not pred_word.startswith("<"):
                words[pos] = pred_word

    return " ".join(words)


@torch.no_grad()
def _correct_with_ssm(text: str, low_pos: list[int], lm_model, lm_vocab, seq_len: int, device: str) -> str:
    words = text.split()
    if len(words) < 2:
        return text

    unk_id = lm_vocab.stoi.get("<unk>", 0)
    ids = [lm_vocab.stoi.get(w, unk_id) for w in words]

    x = ids[:-1]
    xt = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
    logits = lm_model(xt).squeeze(0)

    for pos in low_pos:
        if pos == 0 or pos - 1 >= logits.size(0):
            continue
        pred_id = int(logits[pos - 1].argmax().item())
        pred_word = lm_vocab.itos[pred_id]
        if not pred_word.startswith("<"):
            words[pos] = pred_word

    return " ".join(words)


def _compute_metrics(pred: str, target: str) -> dict[str, float]:
    target = target[: len(pred)]
    return {
        "char_accuracy": character_accuracy(pred, target),
        "word_accuracy": word_accuracy(pred, target),
        "levenshtein": float(levenshtein_distance(pred, target)),
        "bleu": corpus_bleu(pred, target),
        "rouge_l": rouge_l_f1(pred, target),
    }


def main(config_path: str, mode: str) -> None:
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    output_dirs = ensure_output_dirs(config["output"]["base_dir"])

    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    plain = read_plain_text(config["data"]["data_dir"])
    cipher_clean = read_cipher_tokens("cipher_00.txt", config["data"]["data_dir"])
    chars = list(plain)[: len(cipher_clean)]
    cipher_clean = cipher_clean[: len(chars)]

    cipher_vocab = build_vocab(cipher_clean)
    char_vocab = build_vocab(chars)
    lm_vocab = build_vocab(plain.split(), add_mask=True)

    dec_model = DecryptionModel(
        input_vocab_size=len(cipher_vocab.itos),
        output_vocab_size=len(char_vocab.itos),
        embedding_dim=int(config["decryption_model"]["embedding_dim"]),
        hidden_size=int(config["decryption_model"]["hidden_size"]),
        dropout=float(config["decryption_model"]["dropout"]),
        cell_type=config["decryption_model"]["cell_type"],
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

    dec_local = config["decryption_model"]["checkpoint_path"]
    dec_file = config["decryption_model"].get("hf_filename", Path(dec_local).name)
    dec_ckpt = _resolve_checkpoint(config["decryption_model"].get("hf", {}), dec_local, dec_file)
    load_checkpoint(dec_ckpt, dec_model, optimizer=None, device=device)

    lm_local = config["language_model"]["checkpoint_path"]
    lm_file = config["language_model"].get("hf_filename", Path(lm_local).name)
    lm_ckpt = _resolve_checkpoint(config["language_model"].get("hf", {}), lm_local, lm_file)
    load_checkpoint(lm_ckpt, lm_model, optimizer=None, device=device)

    noisy_files = config["data"].get("noisy_files", ["cipher_01.txt", "cipher_02.txt", "cipher_03.txt", "cipher_04.txt"])
    seq_len = int(config["data"].get("seq_len", 128))
    conf_threshold = float(config.get("confidence_threshold", 0.6))

    sections = [f"pipeline=task3_{lm_type}", f"mode={mode}"]

    for filename in noisy_files:
        cipher_noisy = read_cipher_tokens(filename, config["data"]["data_dir"])
        pred_raw, conf = _decrypt_text(dec_model, cipher_noisy, cipher_vocab, char_vocab, seq_len=seq_len, device=device)

        low_pos = _find_low_conf_word_positions(pred_raw, conf, conf_threshold)

        if lm_type == "bilstm":
            pred_corrected = _correct_with_bilstm(pred_raw, low_pos, lm_model, lm_vocab, seq_len=int(config["language_model"].get("seq_len", 32)), device=device)
        else:
            pred_corrected = _correct_with_ssm(pred_raw, low_pos, lm_model, lm_vocab, seq_len=int(config["language_model"].get("seq_len", 32)), device=device)

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
                f"corrected_char_accuracy={m_corr['char_accuracy']:.6f}",
                f"corrected_word_accuracy={m_corr['word_accuracy']:.6f}",
                f"corrected_levenshtein={m_corr['levenshtein']:.0f}",
                f"corrected_bleu={m_corr['bleu']:.6f}",
                f"corrected_rouge_l={m_corr['rouge_l']:.6f}",
            ]
        )

    output_file = config["output"].get("result_file", f"task3_{lm_type}.txt")
    result_path = Path(output_dirs["results"]) / output_file
    write_text(str(result_path), "\n".join(sections))

    input_file = config.get("input_file")
    output_text_file = config.get("output_text_file")
    if input_file and output_text_file:
        cipher_custom = read_cipher_tokens(input_file, config["data"]["data_dir"])
        pred_raw, conf = _decrypt_text(dec_model, cipher_custom, cipher_vocab, char_vocab, seq_len=seq_len, device=device)
        low_pos = _find_low_conf_word_positions(pred_raw, conf, conf_threshold)
        if lm_type == "bilstm":
            pred_custom = _correct_with_bilstm(pred_raw, low_pos, lm_model, lm_vocab, seq_len=int(config["language_model"].get("seq_len", 32)), device=device)
        else:
            pred_custom = _correct_with_ssm(pred_raw, low_pos, lm_model, lm_vocab, seq_len=int(config["language_model"].get("seq_len", 32)), device=device)
        write_text(output_text_file, pred_custom)
