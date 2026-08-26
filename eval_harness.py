"""
Evaluation / weight-tuning harness against the Mendeley dataset's ground
truth (Teacher_manual_marks_Anonymized.csv).

Two things this does that "does it run" testing can't:

1. Reports how close the pipeline's scores are to a real instructor's
   marks (MAE, RMSE, Pearson r), split by MCQ vs short-answer, so "does
   the calibration layer actually help" has a number attached instead of
   being a hunch.

2. Grid-searches the three short-answer blend weights (embedding / LLM /
   rubric) to find whichever combination best matches the teacher's marks
   on your data, instead of trusting the fixed 1/3-1/3-1/3 split
   short_answer_grader.py ships with.

The grid search is deliberately decoupled from re-calling the LLM: run the
pipeline once, cache each question's three raw signal scores per student
as a RawQuestionSignals, and the search just re-combines those cached
numbers — no repeat API spend for what's actually a cheap search over
~66 weight combinations.

Expected CSV shape (adjust `student_id_col` / the "Q<n>" column naming
below if your copy differs — the dataset page describes the file as
covering "Q1 through Q35" per student row but doesn't give exact headers):

    student_id, Q1, Q2, ..., Q20, Q21, ..., Q35
    Student_1,  1,  0, ...,  1,   1.5, ...,  2

As with mendeley_loader.py, this was written from the dataset's
documented structure, not the real file — check load_ground_truth's
output against a few rows of the actual CSV before trusting it.
"""

import csv
import statistics
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

from mendeley_loader import MCQ_QIDS, SHORT_ANSWER_QIDS

MCQ_MAX_MARKS = 1
SHORT_ANSWER_MAX_MARKS = 2


@dataclass
class RawQuestionSignals:
    """Cache of the three independent short-answer signals for one
    student/question, so weight tuning never has to re-call the LLM."""
    student_id: str
    qid: str
    embedding_score: int
    llm_score: float
    rubric_score: int
    teacher_marks: float


def load_ground_truth(csv_path: str, student_id_col: str = "student_id") -> Dict[Tuple[str, str], float]:
    """Returns {(student_id, qid): marks} from the teacher's CSV."""
    ground_truth = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            student_id = row[student_id_col]
            for qid in MCQ_QIDS + SHORT_ANSWER_QIDS:
                col = f"Q{qid}"
                if col in row and row[col] not in ("", None):
                    ground_truth[(student_id, qid)] = float(row[col])
    return ground_truth


def _error_metrics(predicted: List[float], actual: List[float]) -> Dict[str, float]:
    if not predicted:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson_r": float("nan"), "n": 0}

    errors = [p - a for p, a in zip(predicted, actual)]
    mae = statistics.fmean(abs(e) for e in errors)
    rmse = statistics.fmean(e ** 2 for e in errors) ** 0.5

    if len(predicted) > 1 and statistics.pstdev(predicted) > 0 and statistics.pstdev(actual) > 0:
        r = statistics.correlation(predicted, actual)
    else:
        r = float("nan")

    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "pearson_r": round(r, 3), "n": len(predicted)}


def evaluate_mcq(predicted_marks: Dict[Tuple[str, str], float], ground_truth: Dict[Tuple[str, str], float]) -> Dict:
    keys = [k for k in predicted_marks if k[1] in MCQ_QIDS and k in ground_truth]
    return _error_metrics([predicted_marks[k] for k in keys], [ground_truth[k] for k in keys])


def evaluate_short_answer(predicted_marks: Dict[Tuple[str, str], float], ground_truth: Dict[Tuple[str, str], float]) -> Dict:
    keys = [k for k in predicted_marks if k[1] in SHORT_ANSWER_QIDS and k in ground_truth]
    return _error_metrics([predicted_marks[k] for k in keys], [ground_truth[k] for k in keys])


def tune_short_answer_weights(
    signals: List[RawQuestionSignals],
    step: float = 0.1,
) -> Tuple[Tuple[float, float, float], Dict]:
    """Grid-search (embedding, llm, rubric) weights (summing to 1, in `step`
    increments) to minimize MAE against teacher marks. Returns the best
    weights and their metrics."""
    best_weights, best_metrics, best_mae = None, None, float("inf")

    steps = [round(i * step, 2) for i in range(int(round(1 / step)) + 1)]
    for w_emb, w_llm in product(steps, repeat=2):
        w_rub = round(1 - w_emb - w_llm, 2)
        if w_rub < 0 or w_rub > 1:
            continue

        predicted, actual = [], []
        for s in signals:
            blended = w_emb * s.embedding_score + w_llm * s.llm_score + w_rub * s.rubric_score
            predicted.append(blended / 100 * SHORT_ANSWER_MAX_MARKS)
            actual.append(s.teacher_marks)

        metrics = _error_metrics(predicted, actual)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_weights = (w_emb, w_llm, w_rub)
            best_metrics = metrics

    return best_weights, best_metrics
