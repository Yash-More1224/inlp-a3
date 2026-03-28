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


def rouge_l_f1(pred: str, target: str, chunk_words: int = 2000) -> float:
    """ROUGE-L F1 using chunked difflib.SequenceMatcher.

    rouge_score's ROUGE-L builds Python-list LCS tables: even at 5K-word
    chunks that is 5K*5K*28 bytes ~700 MB per chunk, and Python GC does not
    release between iterations -> OOM.
    difflib.SequenceMatcher is C-backed and uses O(N) memory (Ratcliff-
    Obershelp matching blocks, no explicit matrix), so it never OOMs.
    We accumulate LCS hits, p_len, t_len across chunks and compute global F1.
    """
    import difflib

    p_words = pred.split()
    t_words = target.split()
    if not p_words or not t_words:
        return 0.0

    total_lcs = 0
    total_p = 0
    total_t = 0

    max_len = max(len(p_words), len(t_words))
    for i in range(0, max_len, chunk_words):
        p_chunk = p_words[i : i + chunk_words]
        t_chunk = t_words[i : i + chunk_words]
        if not p_chunk or not t_chunk:
            continue
        sm = difflib.SequenceMatcher(None, p_chunk, t_chunk, autojunk=False)
        total_lcs += sum(block.size for block in sm.get_matching_blocks())
        total_p += len(p_chunk)
        total_t += len(t_chunk)

    precision = total_lcs / max(1, total_p)
    recall = total_lcs / max(1, total_t)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
