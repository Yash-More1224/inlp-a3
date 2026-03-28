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


def levenshtein_distance(a: str, b: str, chunk_size: int = 10000) -> int:
    """Calculate Levenshtein distance in chunks to avoid O(N^2) bottlenecks on massive strings."""
    total_dist = 0
    max_len = max(len(a), len(b))
    
    for i in range(0, max_len, chunk_size):
        a_chunk = a[i:i + chunk_size]
        b_chunk = b[i:i + chunk_size]
        total_dist += _lev.distance(a_chunk, b_chunk)
        
    return total_dist


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


def rouge_l_f1(pred: str, target: str, chunk_words: int = 5000) -> float:
    """ROUGE-L F1 computed in word-chunks to avoid O(N²) LCS memory on large texts.

    rouge_score's ROUGE-L allocates a full LCS DP matrix (N×M words).
    For 514K words that is ~2 TB of RAM and causes OOM.
    Instead we split into chunks of `chunk_words` words, compute per-chunk
    ROUGE-L F1, then return the (weighted) average.
    """
    p_words = pred.split()
    t_words = target.split()
    if not p_words or not t_words:
        return 0.0

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        scores, weights = [], []
        max_len = max(len(p_words), len(t_words))
        for i in range(0, max_len, chunk_words):
            p_chunk = " ".join(p_words[i : i + chunk_words])
            t_chunk = " ".join(t_words[i : i + chunk_words])
            if not p_chunk or not t_chunk:
                continue
            result = scorer.score(t_chunk, p_chunk)
            chunk_len = max(
                len(p_words[i : i + chunk_words]),
                len(t_words[i : i + chunk_words]),
            )
            scores.append(result["rougeL"].fmeasure)
            weights.append(chunk_len)

        if not scores:
            return 0.0
        total_weight = sum(weights)
        return float(sum(s * w for s, w in zip(scores, weights)) / total_weight)

    except Exception:
        # Fallback: fast token-overlap F1 (no LCS, no memory pressure)
        p = set(p_words)
        t = set(t_words)
        if not p or not t:
            return 0.0
        overlap = len(p & t)
        precision = overlap / max(1, len(p_words))
        recall = overlap / max(1, len(t_words))
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))

