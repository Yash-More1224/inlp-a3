from __future__ import annotations

import math

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
    """BLEU score using sacrebleu (Cython-backed) — orders of magnitude faster than
    a pure-Python Counter loop on large corpora (e.g. 514K tokens)."""
    if not pred.strip() or not target.strip():
        return 0.0
    try:
        import sacrebleu as _sb
        result = _sb.corpus_bleu([pred], [[target]], tokenize="none")
        return float(result.score) / 100.0  # sacrebleu returns 0-100
    except Exception:
        # Fallback: fast pure-Python unigram precision only
        from collections import Counter
        p = pred.split()
        t = Counter(target.split())
        if not p:
            return 0.0
        hits = sum(min(1, t[w]) for w in p)
        return hits / max(1, len(p))


def rouge_l_f1(pred: str, target: str) -> float:
    """ROUGE-L F1 using the rouge_score library (C-backed) — avoids the O(N²)
    worst-case of difflib.SequenceMatcher on 500K-word lists."""
    if not pred.strip() or not target.strip():
        return 0.0
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        result = scorer.score(target, pred)
        return float(result["rougeL"].fmeasure)
    except Exception:
        # Fallback: fast token overlap F1
        p = set(pred.split())
        t = set(target.split())
        if not p or not t:
            return 0.0
        overlap = len(p & t)
        precision = overlap / max(1, len(p))
        recall = overlap / max(1, len(t))
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))
