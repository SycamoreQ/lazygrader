"""
LLM-based calibration layer for the answer evaluation pipeline.

Three things happen here that a plain embedding-similarity score can't do:

1. Per-question grading. The sheet is segmented (segmenter.py) into
   individual questions and each is judged on its own, so one badly
   answered question can't get diluted into a passing sheet-level average,
   and a strong answer elsewhere can't mask it either.

2. Self-consistency judging. Each question is judged N_SAMPLES times at a
   nonzero temperature instead of once. The *spread* across samples is a
   real uncertainty signal: a question the judge scores consistently
   (82, 85, 80) is a different situation from one where it swings wildly
   (40, 75, 55), even when the means happen to match — only the second
   should get flagged for a human to look at.

3. Evidence-grounded explanations. The judge is asked to quote the actual
   phrases in the model/student answers that drove its score, instead of
   producing free-floating prose that sounds plausible but can't be
   checked against the source text.

Each question's final score is a 50/50 blend of its own embedding-
similarity score and its mean LLM score; the document-level score is the
average across questions.
"""

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional
import time
from segmenter import QuestionSegment


DEFAULT_CALIBRATION_MODEL = "mistral-large-latest"
N_SAMPLES = 3                    # self-consistency: independent judge calls per question
JUDGE_TEMPERATURE = 0.7          # nonzero so the samples can actually disagree
DISAGREEMENT_THRESHOLD = 20      # points: |embedding_score - mean_llm_score| beyond this -> review
UNCERTAINTY_STD_THRESHOLD = 12   # points: stdev across self-consistency samples beyond this -> review
SIMILARITY_WEIGHT = 0.5
LLM_WEIGHT = 0.5
MAX_EVIDENCE_ITEMS = 5


@dataclass
class QuestionResult:
    qid: str
    embedding_score: int
    llm_scores: List[int] = field(default_factory=list)      # raw self-consistency samples
    llm_score_mean: Optional[float] = None
    llm_score_std: Optional[float] = None
    final_score: int = 0
    reasoning: str = ""
    matched_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    discrepancy: Optional[float] = None
    uncertain: bool = False         # high spread across self-consistency samples
    needs_review: bool = False      # uncertain OR judge/embedding disagree OR calibration failed
    calibration_status: str = "ok"  # "ok" | "unavailable" | "parse_error"


@dataclass
class CalibrationResult:
    final_score: int
    embedding_score: int
    questions: List[QuestionResult]
    reasoning: str
    needs_review: bool
    calibration_status: str         # "ok" | "partial" (some questions degraded)


JUDGE_PROMPT = """You are an impartial exam grader.

You will be shown a MODEL ANSWER (reference/correct answer) and a STUDENT ANSWER
(extracted via OCR, so it may contain minor transcription errors — do not penalize for those)
for a single question.

MODEL ANSWER:
{model_answer}

STUDENT ANSWER:
{student_answer}

Judge the student answer on its actual content and reasoning, not just surface wording
overlap. Ground your judgment in the text: quote short exact phrases (a few words each)
from the two answers to support your score.

Return ONLY a JSON object with this exact shape, nothing else:
{{
  "score": <integer 0-100>,
  "reasoning": "<2-3 sentence explanation of the score, written for the student>",
  "matched_evidence": ["<short exact phrase from the student answer that matches the model answer>", "..."],
  "missing_evidence": ["<short exact phrase or idea from the model answer the student answer lacks>", "..."]
}}
"""


class LLMCalibrator:
    """Wraps a Mistral chat model as a self-consistent, evidence-grounded judge."""

    def __init__(self, client, model=DEFAULT_CALIBRATION_MODEL, n_samples=N_SAMPLES):
        self.client = client
        self.model = model
        self.n_samples = n_samples

    def _ask_llm_once(self, model_answer, student_answer):
        prompt = JUDGE_PROMPT.format(model_answer=model_answer, student_answer=student_answer)
        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=JUDGE_TEMPERATURE,
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse(raw_text):
        cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        score = max(0, min(100, int(parsed["score"])))
        return {
            "score": score,
            "reasoning": str(parsed.get("reasoning", "")).strip(),
            "matched_evidence": [str(e).strip() for e in parsed.get("matched_evidence", [])][:MAX_EVIDENCE_ITEMS],
            "missing_evidence": [str(e).strip() for e in parsed.get("missing_evidence", [])][:MAX_EVIDENCE_ITEMS],
        }

    def calibrate_question(self, segment: QuestionSegment, embedding_score: int) -> QuestionResult:
        """Judge one question N_SAMPLES times and aggregate. Never raises —
        falls back to the embedding score alone if every sample fails."""
        samples = []
        for _ in range(self.n_samples):
            try:
                raw = self._ask_llm_once(segment.model_text, segment.student_text)
                samples.append(self._parse(raw))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
            except Exception:
                continue

        if not samples:
            return QuestionResult(
                qid=segment.qid,
                embedding_score=embedding_score,
                final_score=embedding_score,
                reasoning="Calibration model unavailable or unparseable for this question; "
                          "used embedding score only.",
                needs_review=True,
                calibration_status="unavailable",
            )

        scores = [s["score"] for s in samples]
        mean_score = statistics.fmean(scores)
        std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0

        # Representative sample = whichever run's score is closest to the
        # mean, so reported reasoning reflects a typical run, not a
        # cherry-picked best (or worst) case.
        representative = min(samples, key=lambda s: abs(s["score"] - mean_score))

        final_score = round(SIMILARITY_WEIGHT * embedding_score + LLM_WEIGHT * mean_score)
        discrepancy = abs(embedding_score - mean_score)
        uncertain = std_score > UNCERTAINTY_STD_THRESHOLD

        def _dedupe(items):
            seen, out = set(), []
            for item in items:
                key = item.casefold()
                if key and key not in seen:
                    seen.add(key)
                    out.append(item)
            return out[:MAX_EVIDENCE_ITEMS]

        matched = _dedupe(e for s in samples for e in s["matched_evidence"])
        missing = _dedupe(e for s in samples for e in s["missing_evidence"])

        return QuestionResult(
            qid=segment.qid,
            embedding_score=embedding_score,
            llm_scores=scores,
            llm_score_mean=round(mean_score, 1),
            llm_score_std=round(std_score, 1),
            final_score=final_score,
            reasoning=representative["reasoning"],
            matched_evidence=matched,
            missing_evidence=missing,
            discrepancy=round(discrepancy, 1),
            uncertain=uncertain,
            needs_review=uncertain or discrepancy > DISAGREEMENT_THRESHOLD,
            calibration_status="ok" if len(samples) == self.n_samples else "parse_error",
        )

    def calibrate_document(self, segments: List[QuestionSegment], embedding_scores: List[int]) -> CalibrationResult:
        """Grade every question segment and aggregate into a document-level result."""
        questions = [
            self.calibrate_question(seg, emb_score)
            for seg, emb_score in zip(segments, embedding_scores)
        ]

        final_score = round(statistics.fmean(q.final_score for q in questions))
        embedding_score = round(statistics.fmean(q.embedding_score for q in questions))
        needs_review = any(q.needs_review for q in questions)
        calibration_status = "ok" if all(q.calibration_status == "ok" for q in questions) else "partial"

        if len(questions) == 1 and questions[0].qid == "full":
            reasoning = questions[0].reasoning
        else:
            reasoning = "\n".join(f"Q{q.qid}: {q.reasoning}" for q in questions)

        return CalibrationResult(
            final_score=final_score,
            embedding_score=embedding_score,
            questions=questions,
            reasoning=reasoning,
            needs_review=needs_review,
            calibration_status=calibration_status,
        )
    
    def _ask_llm_with_retry(self, model_answer, student_answer, max_retries=3):
        delay = 2

        for attempt in range(max_retries + 1):
            try:
                return self._ask_llm_once(model_answer, student_answer)

            except Exception as exc:
                error_text = str(exc).lower()

                is_rate_limit = (
                    "429" in error_text
                    or "rate limit" in error_text
                    or "too many requests" in error_text
                )

                if not is_rate_limit or attempt == max_retries:
                    raise

                print(
                    f"[CALIBRATOR] Rate limited. "
                    f"Retry {attempt + 1}/{max_retries} in {delay}s..."
                )

                time.sleep(delay)
                delay *= 2