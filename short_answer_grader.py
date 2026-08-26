"""
Combines three independent signals for short-answer grading (Q21-Q35):
SBERT embedding similarity, the self-consistency LLM judge (calibrator.py),
and rubric keypoint coverage (rubric_scorer.py).
"""

from dataclasses import dataclass

from calibrator import LLMCalibrator, QuestionResult as LLMQuestionResult
from mendeley_loader import AnswerKeyEntry
from rubric_scorer import RubricResult, score_rubric
from segmenter import QuestionSegment

EMBEDDING_WEIGHT = 1 / 3
LLM_WEIGHT = 1 / 3
RUBRIC_WEIGHT = 1 / 3
OUTLIER_THRESHOLD = 25  # points: any one signal this far from the blended score -> review


@dataclass
class ShortAnswerResult:
    qid: str
    embedding_score: int
    llm_result: LLMQuestionResult
    rubric_result: RubricResult
    final_score: int
    needs_review: bool
    reasoning: str


def grade_short_answer(
    segment: QuestionSegment,
    key_entry: AnswerKeyEntry,
    embedding_score: int,
    calibrator: LLMCalibrator,
) -> ShortAnswerResult:
    llm_result = calibrator.calibrate_question(segment, embedding_score)
    return combine_short_answer_signals(segment, key_entry, embedding_score, llm_result)


def combine_short_answer_signals(
    segment: QuestionSegment,
    key_entry: AnswerKeyEntry,
    embedding_score: int,
    llm_result: LLMQuestionResult,
) -> ShortAnswerResult:
    rubric_result = score_rubric(key_entry.keypoints, segment.student_text)

    llm_score = llm_result.llm_score_mean if llm_result.llm_score_mean is not None else embedding_score

    final_score = round(
        EMBEDDING_WEIGHT * embedding_score
        + LLM_WEIGHT * llm_score
        + RUBRIC_WEIGHT * rubric_result.score
    )

    signals = [embedding_score, llm_score, rubric_result.score]
    outlier = any(abs(s - final_score) > OUTLIER_THRESHOLD for s in signals)

    return ShortAnswerResult(
        qid=segment.qid,
        embedding_score=embedding_score,
        llm_result=llm_result,
        rubric_result=rubric_result,
        final_score=final_score,
        needs_review=llm_result.needs_review or outlier,
        reasoning=llm_result.reasoning,
    )