from __future__ import annotations

import difflib
import math
from collections import Counter

import Levenshtein as _lev


def character_accuracy(pred: str, target: str) -> float:
    length = min(len(pred), len(target))
    if length == 0:
        return 0.0
    correct = sum(1 for p, t in zip(pred[:length], target[:length]) if p == t)
    return correct / length


def word_accuracy(pred: str, target: str) -> float:
    p_words = pred.split()
    t_words = target.split()
    length = min(len(p_words), len(t_words))
    if length == 0:
        return 0.0
    correct = sum(1 for p, t in zip(p_words[:length], t_words[:length]) if p == t)
    return correct / length


def levenshtein_distance(a: str, b: str) -> int:
    """Calculate Levenshtein distance between two strings using python-Levenshtein library (C extension)."""
    return _lev.distance(a, b)


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(min(50.0, loss)))


def corpus_bleu(pred: str, target: str, max_n: int = 4) -> float:
    pred_tokens = pred.split()
    target_tokens = target.split()
    if not pred_tokens or not target_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(max(0, len(pred_tokens) - n + 1)))
        tgt_ngrams = Counter(tuple(target_tokens[i : i + n]) for i in range(max(0, len(target_tokens) - n + 1)))
        if not pred_ngrams:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, tgt_ngrams[ng]) for ng, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        precisions.append(max(overlap / max(1, total), 1e-9))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(pred_tokens) > len(target_tokens) else math.exp(1 - (len(target_tokens) / max(1, len(pred_tokens))))
    return float(bp * geo_mean)


def rouge_l_f1(pred: str, target: str) -> float:
    """ROUGE-L F1 using difflib.SequenceMatcher (C-backed LCS), much faster than pure-Python DP."""
    p = pred.split()
    t = target.split()
    if not p or not t:
        return 0.0

    # SequenceMatcher uses Ratcliff/Obershelp matching which finds matching blocks
    # equivalent to LCS for our purposes (no autojunk to keep it exact).
    sm = difflib.SequenceMatcher(None, p, t, autojunk=False)
    lcs = sum(block.size for block in sm.get_matching_blocks())

    precision = lcs / max(1, len(p))
    recall = lcs / max(1, len(t))
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))
