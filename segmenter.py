"""
Splits an OCR'd answer sheet into per-question segments so grading can
happen question-by-question instead of on the whole sheet as one blob.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class QuestionSegment:
    qid: str
    model_text: str
    student_text: str


# Matches line-starting headers like "Q1.", "Q.1)", "Question 1:", "**Q1)**".
# Deliberately conservative: under-splitting just falls back to whole-sheet
_QUESTION_HEADER = re.compile(
    r'(?im)^[ \t]*(?:\*\*)?[ \t]*Q(?:uestion)?[ \t.]*[-]?[ \t]*(\d+)[ \t]*[\.\):]?[ \t]*(?:\*\*)?[ \t]*[:\-]?[ \t]*'
)


def split_by_question_header(text: str) -> Dict[str, str]:
    """Split text into {question_number: body_text}. Returns {} if fewer
    than 2 headers are found — i.e. no clear per-question structure.

    Public so other modules (mendeley_loader.py) can reuse the same
    header convention instead of re-implementing question detection."""
    matches = list(_QUESTION_HEADER.finditer(text))
    if len(matches) < 2:
        return {}

    segments: Dict[str, str] = {}
    for i, m in enumerate(matches):
        qid = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if qid not in segments or len(body) > len(segments[qid]):
            segments[qid] = body
    return segments


def segment_answers(model_text: str, student_text: str) -> List[QuestionSegment]:
    model_segments = split_by_question_header(model_text)
    student_segments = split_by_question_header(student_text)

    if not model_segments or not student_segments:
        return [QuestionSegment(qid="full", model_text=model_text, student_text=student_text)]

    segments = []
    for qid in sorted(model_segments, key=int):
        segments.append(
            QuestionSegment(
                qid=qid,
                model_text=model_segments[qid],
                # Missing on the student side means they skipped it — grade
                # that honestly instead of silently dropping the question.
                student_text=student_segments.get(qid, "").strip() or "[No answer provided]",
            )
        )
    return segments


_BARE_NUMBER_HEADER = re.compile(r'(?m)^[ \t]*(\d{1,2})[ \t]*[.\):]?[ \t]+')


def split_by_known_qids(text: str, valid_qids) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    valid = set(valid_qids)
    matches = [(m.group(1), m) for m in _BARE_NUMBER_HEADER.finditer(text) if m.group(1) in valid]

    segments: Dict[str, str] = {}
    leftovers: List[Tuple[str, str]] = []
    seen = set()

    for i, (qid, m) in enumerate(matches):
        start = m.end()
        end = matches[i + 1][1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        if qid not in seen:
            seen.add(qid)
            segments[qid] = body
        else:
            leftovers.append((qid, body))

    return segments, leftovers