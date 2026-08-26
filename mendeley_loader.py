import csv
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from segmenter import split_by_question_header


MCQ_QIDS = [str(i) for i in range(1, 21)]            # Q1-Q20
SHORT_ANSWER_QIDS = [str(i) for i in range(21, 36)]  # Q21-Q35

_KEYPOINT_SPLIT = re.compile(r'[.;]')

_ABBREVIATIONS = [
    (re.compile(r'\be\.g\.,?', re.IGNORECASE), 'for example'),
    (re.compile(r'\bi\.e\.,?', re.IGNORECASE), 'that is'),
    (re.compile(r'\betc\.', re.IGNORECASE), 'and so on'),
]

_HAS_CONTENT = re.compile(r'[A-Za-z]{3,}')


def _split_keypoints(text: str) -> List[str]:
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    raw = [kp.strip() for kp in _KEYPOINT_SPLIT.split(text)]
    return [kp for kp in raw if _HAS_CONTENT.search(kp)]


@dataclass
class AnswerKeyEntry:
    qid: str
    qtype: str                                   # "mcq" | "short"
    correct_option: Optional[str] = None          # MCQ only
    model_answer: str = ""                        # short-answer only: full rubric text
    keypoints: List[str] = field(default_factory=list)  # short-answer only


def load_answer_key(path: str) -> Dict[str, AnswerKeyEntry]:
    entries: Dict[str, AnswerKeyEntry] = {}

    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["Question_Number"]).strip()
            qtype = str(row["Type"]).strip().lower()
            answer = str(row["Correct_Answer"]).strip()

            if qtype == "mcq":
                match = re.search(r'\b([A-D])\b', answer)
                entries[qid] = AnswerKeyEntry(
                    qid=qid,
                    qtype="mcq",
                    correct_option=match.group(1) if match else None,
                )
            else:
                keypoints = _split_keypoints(answer)
                entries[qid] = AnswerKeyEntry(
                    qid=qid,
                    qtype="short",
                    model_answer=answer,
                    keypoints=keypoints if keypoints else [answer],
                )

    return entries


def load_questions(path: str) -> Dict[str, str]:
    """Question.txt -> {qid: question_text}. Tries a CSV read first (same
    style as answerkey.txt); falls back to segmenter.py's prose-header
    splitter if the file isn't CSV-shaped.
    """
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        raw = f.read()

    try:
        reader = csv.DictReader(raw.splitlines())
        if reader.fieldnames and any("question" in (fn or "").lower() for fn in reader.fieldnames):
            number_col = next(fn for fn in reader.fieldnames if "number" in fn.lower())
            text_col = next(fn for fn in reader.fieldnames if "text" in fn.lower() or "question" == fn.lower())
            return {str(row[number_col]).strip(): str(row[text_col]).strip() for row in reader}
    except (StopIteration, KeyError, csv.Error):
        pass

    return split_by_question_header(raw)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python mendeley_loader.py <path-to-answerkey.txt>")
        raise SystemExit(1)

    key = load_answer_key(sys.argv[1])
    if not key:
        print("No rows parsed — check that the file has Question_Number,Type,Correct_Answer columns.")
        raise SystemExit(1)

    missing_mcq = [q for q in MCQ_QIDS if q not in key]
    missing_short = [q for q in SHORT_ANSWER_QIDS if q not in key]
    if missing_mcq or missing_short:
        print(f"WARNING: missing questions — MCQ: {missing_mcq or 'none'}, "
              f"short-answer: {missing_short or 'none'}")
        print()

    for qid in sorted(key, key=int):
        e = key[qid]
        if e.qtype == "mcq":
            flag = "  <-- no option detected, check the Correct_Answer format" if e.correct_option is None else ""
            print(f"Q{qid} [MCQ]   correct_option={e.correct_option!r}{flag}")
        else:
            print(f"Q{qid} [SHORT] {len(e.keypoints)} keypoint(s): {e.keypoints}")
