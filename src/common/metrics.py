from __future__ import annotations

import math
from collections import Counter


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
    """Calculate Levenshtein distance between two strings."""
    try:
        import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        # Fallback implementation if python-Levenshtein is not installed
        if len(a) < len(b):
            return levenshtein_distance(b, a)
        
        if len(b) == 0:
            return len(a)
        
        previous_row = range(len(b) + 1)
        for i, c1 in enumerate(a):
            current_row = [i + 1]
            for j, c2 in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


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
    p = pred.split()
    t = target.split()
    if not p or not t:
        return 0.0

    dp = [[0] * (len(t) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(t) + 1):
            if p[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / max(1, len(p))
    recall = lcs / max(1, len(t))
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))
