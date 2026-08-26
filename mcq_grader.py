"""
Deterministic grading for the 20 multiple-choice questions (Q1-Q20 in the
Mendeley dataset's numbering).

No LLM call needed here: once OCR has extracted the text, the student's
marked option either matches the answer key or it doesn't. Sending this to
an LLM (or even to embedding similarity) would add cost, latency, and a
little noise for zero benefit over exact comparison.

The one place this genuinely needs care is knowing when it *can't* tell
what the student marked (garbled OCR, no option detected) — that's a
different failure mode from "marked the wrong option", and gets flagged
`needs_review` instead of silently scored as incorrect.
"""

import re
from dataclasses import dataclass
from typing import Optional

from mendeley_loader import AnswerKeyEntry


# Matches the student's marked option in common OCR'd forms:
# "B", "(B)", "Ans: B", "Answer - B", "Option B", "B)"
_STUDENT_OPTION = re.compile(
    r'(?:ans(?:wer)?\s*[:\-]?\s*|option\s*)?\(?\b([A-D])\b\)?',
    re.IGNORECASE,
)


@dataclass
class MCQResult:
    qid: str
    correct_option: Optional[str]
    student_option: Optional[str]
    is_correct: bool
    needs_review: bool   # True when no option could be confidently read
    score: int            # 0 or 100, same 0-100 scale as the rest of the pipeline


def grade_mcq(qid: str, student_text: str, key_entry: AnswerKeyEntry) -> MCQResult:
    match = _STUDENT_OPTION.search(student_text.strip())
    student_option = match.group(1).upper() if match else None

    if student_option is None or key_entry.correct_option is None:
        return MCQResult(
            qid=qid,
            correct_option=key_entry.correct_option,
            student_option=student_option,
            is_correct=False,
            needs_review=True,   # couldn't determine an answer -> a human should check the OCR, not us
            score=0,
        )

    is_correct = student_option == key_entry.correct_option
    return MCQResult(
        qid=qid,
        correct_option=key_entry.correct_option,
        student_option=student_option,
        is_correct=is_correct,
        needs_review=False,
        score=100 if is_correct else 0,
    )
