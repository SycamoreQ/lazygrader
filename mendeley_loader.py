"""
Loader for the Mendeley "Digitized Student Examination Papers, Answer Keys,
and Manual Evaluations" dataset:
https://data.mendeley.com/datasets/sf3kvjwknt/1

answerkey.txt is a CSV (the .txt extension is misleading) with columns:

    Question_Number,Type,Correct_Answer
    1,MCQ,B
    21,Short_Answer,"A type of ML where the model is trained on labeled
                      data with known inputs and outputs."

Q1-Q20 are MCQ (Type=MCQ, Correct_Answer is a single option letter).
Q21-Q35 are short answer (Type=Short_Answer, Correct_Answer is the model
answer / rubric text). Rubric text separates distinct concepts with '.'
or ';' and uses ',' for a list of examples *within* one concept (e.g.
"Detection: Z-score, IQR, Scatter plots, Box plots." is ONE keypoint
about detection methods, not four) — so keypoints are split on '.'/';'
only, never on commas.

Expected directory layout (point MENDELEY_DIR in evaluate_mendeley.py at
wherever you unzip the dataset):

    <dataset_root>/
        Question.txt                        # exam questionnaire
        answerkey.txt                        # CSV, see above
        Student_Pdf/                         # raw scanned student sheets
        Corrected_Pdf/                       # teacher-annotated sheets (unused here)
        Teacher_manual_marks_Anonymized.csv  # ground-truth per-question marks

Question.txt's exact format is still unverified (this environment can't
fetch the dataset) — load_questions() tries a plain CSV read first
(Question_Number/Question_Text-style columns) and falls back to
segmenter.py's prose-header splitter if that fails. It's optional: the
grading pipeline only needs answerkey.txt. Run
`python mendeley_loader.py <dataset_dir>/answerkey.txt` before trusting
any of this against the real file.
"""

import csv
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from segmenter import split_by_question_header


MCQ_QIDS = [str(i) for i in range(1, 21)]            # Q1-Q20
SHORT_ANSWER_QIDS = [str(i) for i in range(21, 36)]  # Q21-Q35

# Split rubric text into keypoints on sentence/clause boundaries ('.', ';')
# only — never on ',', since commas are used for example-lists within a
# single concept (see module docstring).
_KEYPOINT_SPLIT = re.compile(r'[.;]')

# Abbreviations whose periods would otherwise look like sentence boundaries
# ("e.g." has two dots and would get shredded into "e", "g", ...). Expand
# to plain words rather than just stripping the dots, so the meaning
# survives for word-overlap matching in rubric_scorer.py too.
_ABBREVIATIONS = [
    (re.compile(r'\be\.g\.,?', re.IGNORECASE), 'for example'),
    (re.compile(r'\bi\.e\.,?', re.IGNORECASE), 'that is'),
    (re.compile(r'\betc\.', re.IGNORECASE), 'and so on'),
]

# A split fragment needs at least one real word to count as a keypoint —
# filters out artifacts like a lone leftover quote mark from splitting
# "...'Senior.'" on its internal period.
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
                # Correct_Answer is expected to be a bare option letter (e.g. "B").
                # Guard against stray formatting ("(B)", "B.") just in case.
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

    Optional — NOT required for grading. evaluate_mendeley.py never calls
    this; it only exists in case you want to hand the actual question
    text to the LLM judge for extra context. If it returns {}, that's not
    blocking anything — but if you do want it working, run:
        python -c "from mendeley_loader import load_questions; print(load_questions('data/Question.txt'))"
    and if it's still {}, paste the first ~10 lines of Question.txt so the
    parser can be written against its real format instead of a guess.
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
