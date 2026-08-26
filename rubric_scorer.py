"""
Light rubric-based scoring for short-answer questions (Q21-Q35): a cheap,
fully deterministic third signal alongside embedding similarity and the
LLM judge. Checks how many of the answer key's rubric keypoints show up in
the student's answer, using significant-word overlap rather than exact
string matching so it survives OCR noise and rewording.

This is intentionally simple 
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to",
    "and", "or", "for", "with", "by", "as", "that", "this", "it", "be",
    "which", "at", "from", "such", "can", "may", "will", "into", "also",
}
_WORD = re.compile(r"[a-zA-Z']+")


def _significant_words(text: str) -> set:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOPWORDS and len(w) > 2}


def _word_weights(keypoints: List[str]) -> dict:
    """IDF-style weight per word across this question's keypoints: a word
    that shows up in most keypoints (e.g. 'training', 'data' in a Data
    Science rubric) is domain noise and shouldn't count as much toward a
    match as a word that's distinctive to just one keypoint (e.g.
    'regularization'). Without this, a student answer that only mentions
    generic domain terms can spuriously "match" a keypoint it never
    actually addressed."""
    doc_freq = Counter()
    for kp in keypoints:
        doc_freq.update(_significant_words(kp))
    n = len(keypoints) or 1
    return {w: math.log((n + 1) / (df + 1)) + 1 for w, df in doc_freq.items()}


@dataclass
class RubricResult:
    matched_keypoints: List[str] = field(default_factory=list)
    missed_keypoints: List[str] = field(default_factory=list)
    coverage: float = 0.0   # fraction of keypoints matched, 0-1
    score: int = 0          # coverage * 100, for blending with the other two signals


def score_rubric(keypoints: List[str], student_text: str, overlap_threshold: float = 0.4) -> RubricResult:
    """`overlap_threshold` still decides whether a keypoint is reported as
    'matched' vs 'missed' (for display/triangulation purposes), but the
    numeric score is NOT a count of keypoints that cleared that threshold.
    """
    student_words = _significant_words(student_text)
    weights = _word_weights(keypoints)
    matched, missed = [], []
    weighted_ratios = []  # (overlap_ratio, keypoint_total_weight)

    for kp in keypoints:
        kp_words = _significant_words(kp)
        if not kp_words:
            continue
        total_weight = sum(weights.get(w, 1.0) for w in kp_words)
        matched_weight = sum(weights.get(w, 1.0) for w in (kp_words & student_words))
        overlap = matched_weight / total_weight if total_weight else 0.0
        weighted_ratios.append((overlap, total_weight))
        (matched if overlap >= overlap_threshold else missed).append(kp)

    if not weighted_ratios:
        return RubricResult()

    total_weight = sum(w for _, w in weighted_ratios)
    coverage = (
        sum(overlap * w for overlap, w in weighted_ratios) / total_weight
        if total_weight else 0.0
    )

    return RubricResult(
        matched_keypoints=matched,
        missed_keypoints=missed,
        coverage=round(coverage, 2),
        score=round(coverage * 100),
    )
