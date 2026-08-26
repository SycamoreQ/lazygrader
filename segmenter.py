"""
Splits an OCR'd answer sheet into per-question segments so grading can
happen question-by-question instead of on the whole sheet as one blob.

This matters because a single similarity/LLM score over an entire multi-
question sheet averages away exactly the information a grader cares about:
"which question did they actually get wrong?" A sheet-level 76/100 hides
whether that's six evenly-mediocre answers or three excellent answers and
one blank one.

Detection is regex-based over common OCR'd markers ('Q1', 'Q.1',
'Question 1:', etc.) rather than another model call — it's cheap, and
segmentation errors are easy to reason about and fix, unlike an LLM
silently mis-splitting a sheet.
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
# grading (see segment_answers below), which is the safe failure mode.
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
        # Same question number seen twice (OCR noise/repeated header) -> keep the longer body.
        if qid not in segments or len(body) > len(segments[qid]):
            segments[qid] = body
    return segments


def segment_answers(model_text: str, student_text: str) -> List[QuestionSegment]:
    """Align model/student answers by question number.

    Falls back to a single whole-text segment (qid='full') if either side
    has no detectable per-question structure, so free-form or single-
    question sheets still grade correctly instead of erroring out.
    """
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
    """Split text into {qid: body} using BARE numeric line-start headers
    ('1.', '21)', '25 ' — no 'Q' prefix), anchored against a known set of
    valid question numbers instead of discovering headers blind.

    Use this (not split_by_question_header above) when you already know
    the exact set of question IDs from an answer key — e.g. the Mendeley
    dataset's student sheets, which number answers plainly with no 'Q'.

    Real scanned/handwritten sheets are noisy in ways that would corrupt a
    naive "just match digits" splitter: page markers ("1/ total 3"),
    garbled OCR table remnants, or a misread date can look exactly like a
    header. Two things make this robust against that instead of corrupted
    by it:

    1. A match only counts if its number is in `valid_qids` — arbitrary
       numbers from noise don't qualify as headers at all.
    2. The FIRST occurrence of each qid becomes that question's segment.
       Every later occurrence of an already-seen number (duplicate
       header, OCR garbage repeating a used number) is still cut out as
       its own chunk — so it can't silently get absorbed into an
       unrelated neighboring segment and pollute it — but it's returned
       separately as a "leftover" instead of overwriting anything.

    Returns (segments, leftovers) where leftovers is a list of
    (duplicate_qid_seen, body_text) for every non-first-occurrence chunk.
    These often aren't garbage — see answer_reconciler.py: a student who
    mislabels a question (writes '29' when they mean '28') produces
    exactly this pattern, and the leftover pool is how that gets a chance
    to be recovered instead of silently lost.
    """
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