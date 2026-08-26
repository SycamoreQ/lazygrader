from pathlib import Path
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from segmenter import QuestionSegment


DEFAULT_CALIBRATION_MODEL = "mistral-large-latest"

N_SAMPLES = 1
JUDGE_TEMPERATURE = 0.7

DISAGREEMENT_THRESHOLD = 20
UNCERTAINTY_STD_THRESHOLD = 12

SIMILARITY_WEIGHT = 0.5
LLM_WEIGHT = 0.5

MAX_EVIDENCE_ITEMS = 5

DEFAULT_REQUESTS_PER_SECOND = 0.2
DEFAULT_MAX_RETRIES = 1
DEFAULT_BATCH_SIZE = 5


class RateLimitError(Exception):
    """Raised when the remote calibration API returns HTTP 429."""


class _RateLimiter:
    """Throttle requests to the configured maximum requests/second."""

    def __init__(self, rps: float = DEFAULT_REQUESTS_PER_SECOND):
        self.min_interval = 1.0 / rps if rps and rps > 0 else 0.0
        self._last_call = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return

        remaining = self.min_interval - (
            time.monotonic() - self._last_call
        )

        if remaining > 0:
            time.sleep(remaining)

        self._last_call = time.monotonic()


@dataclass
class QuestionResult:
    qid: str
    embedding_score: int

    llm_scores: List[int] = field(default_factory=list)
    llm_score_mean: Optional[float] = None
    llm_score_std: Optional[float] = None

    final_score: int = 0
    reasoning: str = ""

    matched_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)

    discrepancy: Optional[float] = None
    uncertain: bool = False
    needs_review: bool = False

    # "ok" | "unavailable" | "parse_error"
    calibration_status: str = "ok"


@dataclass
class CalibrationResult:
    final_score: int
    embedding_score: int
    questions: List[QuestionResult]
    reasoning: str
    needs_review: bool

    # "ok" | "partial"
    calibration_status: str


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


BATCH_JUDGE_PROMPT = """You are an impartial exam grader.

You will be shown several questions from the same student sheet, each with a MODEL ANSWER
(reference/correct answer) and a STUDENT ANSWER (extracted via OCR, so it may contain minor
transcription errors — do not penalize for those).

Grade every question independently: a weak or strong answer to one question must not affect
the score you give another. Ground each judgment in the text: quote short exact phrases (a
few words each) from that question's two answers to support its score.

{questions_block}

Return ONLY a JSON object with this exact shape, nothing else — one entry per question id
shown above, using its exact id string:
{{
  "results": [
    {{
      "qid": "<question id, exactly as given above>",
      "score": <integer 0-100>,
      "reasoning": "<2-3 sentence explanation of the score, written for the student>",
      "matched_evidence": ["<short exact phrase from the student answer that matches the model answer>", "..."],
      "missing_evidence": ["<short exact phrase or idea from the model answer the student answer lacks>", "..."]
    }}
  ]
}}
"""


_QUESTION_BLOCK = """--- Question {qid} ---
MODEL ANSWER:
{model_answer}

STUDENT ANSWER:
{student_answer}
"""


class LLMCalibrator:
    """
    LLM calibration with:
      - request throttling
      - 429 circuit breaker
      - bounded retries
      - batched grading
      - embedding fallback

    IMPORTANT:
    Once HTTP 429 is detected, remote calibration is disabled for the rest
    of the current grading run. No later question will make another API call.
    """

    def __init__(
        self,
        client,
        model=DEFAULT_CALIBRATION_MODEL,
        n_samples=N_SAMPLES,
        requests_per_second=DEFAULT_REQUESTS_PER_SECOND,
        max_retries=DEFAULT_MAX_RETRIES,
        batch_size=DEFAULT_BATCH_SIZE,
    ):
        self.client = client
        self.model = model

        self.n_samples = max(1, int(n_samples))
        self.max_retries = max(0, int(max_retries))
        self.batch_size = max(1, int(batch_size))

        self._rate_limiter = _RateLimiter(requests_per_second)

        # Run-level circuit breaker.
        self._remote_disabled = False
        self._disable_reason: Optional[str] = None

    @property
    def remote_disabled(self) -> bool:
        return self._remote_disabled

    @property
    def disable_reason(self) -> Optional[str]:
        return self._disable_reason

    def reset(self):
        """Reset the circuit breaker for a completely new grading run."""
        self._remote_disabled = False
        self._disable_reason = None
        self._rate_limiter._last_call = 0.0

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        error_text = str(exc).lower()

        return (
            "429" in error_text
            or "rate limit" in error_text
            or "rate_limited" in error_text
            or "too many requests" in error_text
        )

    def _disable_remote_calibration(self, reason: str):
        if not self._remote_disabled:
            self._remote_disabled = True
            self._disable_reason = reason

            print(
                "[CALIBRATOR] Remote calibration disabled for the "
                "remainder of this grading run: "
                f"{reason}"
            )

    def _complete(self, prompt: str) -> str:
        """Perform one API request. 429 immediately trips the circuit breaker."""
        if self._remote_disabled:
            raise RateLimitError(
                self._disable_reason
                or "Remote calibration disabled for this run."
            )

        self._rate_limiter.wait()

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_format={"type": "json_object"},
                temperature=JUDGE_TEMPERATURE,
            )
        except Exception as exc:
            if self._is_rate_limit_error(exc):
                self._disable_remote_calibration(
                    "Mistral returned HTTP 429 rate_limited."
                )
                raise RateLimitError(
                    self._disable_reason
                    or "Mistral returned HTTP 429."
                ) from exc
            raise

        try:
            return str(response.choices[0].message.content)
        except Exception as exc:
            raise ValueError(
                "Mistral response did not contain "
                "choices[0].message.content."
            ) from exc

    def _with_retry(self, call_fn, max_retries=None):
        """
        Retry only transient non-429 errors.

        429 is never retried because the run-level circuit breaker takes over.
        """
        if max_retries is None:
            max_retries = self.max_retries

        delay = 2.0

        for attempt in range(max_retries + 1):
            try:
                return call_fn()

            except RateLimitError:
                raise

            except Exception as exc:
                if self._is_rate_limit_error(exc):
                    self._disable_remote_calibration(
                        "Mistral returned HTTP 429 rate_limited."
                    )
                    raise RateLimitError(
                        self._disable_reason
                        or "Mistral returned HTTP 429."
                    ) from exc

                if attempt >= max_retries:
                    raise

                print(
                    "[CALIBRATOR] Temporary error. "
                    f"Retry {attempt + 1}/{max_retries} "
                    f"in {delay:.1f}s..."
                )

                time.sleep(delay)
                delay *= 2

    def _ask_llm_once(self, model_answer, student_answer):
        prompt = JUDGE_PROMPT.format(
            model_answer=model_answer,
            student_answer=student_answer,
        )
        return self._complete(prompt)

    def _ask_llm_with_retry(
        self,
        model_answer,
        student_answer,
        max_retries=None,
    ):
        return self._with_retry(
            lambda: self._ask_llm_once(
                model_answer,
                student_answer,
            ),
            max_retries=max_retries,
        )

    @staticmethod
    def _parse(raw_text):
        """Parse a single-question JSON response."""
        cleaned = raw_text.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        ).strip()

        parsed = json.loads(cleaned)

        score = max(
            0,
            min(
                100,
                int(parsed["score"]),
            ),
        )

        matched = [
            str(e).strip()
            for e in parsed.get("matched_evidence", [])
            if str(e).strip()
        ][:MAX_EVIDENCE_ITEMS]

        missing = [
            str(e).strip()
            for e in parsed.get("missing_evidence", [])
            if str(e).strip()
        ][:MAX_EVIDENCE_ITEMS]

        return {
            "score": score,
            "reasoning": str(
                parsed.get("reasoning", "")
            ).strip(),
            "matched_evidence": matched,
            "missing_evidence": missing,
        }

    def calibrate_question(
        self,
        segment: QuestionSegment,
        embedding_score: int,
    ) -> QuestionResult:
        """Grade one question; never raises for remote availability failures."""

        if self._remote_disabled:
            return QuestionResult(
                qid=segment.qid,
                embedding_score=embedding_score,
                final_score=embedding_score,
                reasoning=(
                    "Remote calibration disabled for this grading run; "
                    "used embedding score only. "
                    f"Reason: {self._disable_reason or 'unavailable'}"
                ),
                needs_review=True,
                calibration_status="unavailable",
            )

        samples = []
        errors = []

        for _ in range(self.n_samples):
            try:
                raw = self._ask_llm_with_retry(
                    segment.model_text,
                    segment.student_text,
                    max_retries=self.max_retries,
                )
                samples.append(self._parse(raw))

            except RateLimitError as exc:
                errors.append(str(exc))
                break

            except (
                json.JSONDecodeError,
                KeyError,
                ValueError,
                TypeError,
            ) as exc:
                errors.append(
                    f"parse error: {type(exc).__name__}: {exc}"
                )

            except Exception as exc:
                errors.append(
                    f"{type(exc).__name__}: {exc}"
                )

        if not samples:
            cause = (
                "; ".join(dict.fromkeys(errors))
                if errors
                else "no successful judge calls"
            )

            print(
                f"    [CALIBRATOR] Q{segment.qid}: "
                f"using embedding fallback -> {cause}"
            )

            return QuestionResult(
                qid=segment.qid,
                embedding_score=embedding_score,
                final_score=embedding_score,
                reasoning=(
                    "Calibration model unavailable or unparseable; "
                    "used embedding score only. "
                    f"Cause: {cause}"
                ),
                needs_review=True,
                calibration_status="unavailable",
            )

        result = self._aggregate_samples(
            segment.qid,
            embedding_score,
            samples,
        )

        if len(samples) < self.n_samples:
            result.calibration_status = "parse_error"
            result.needs_review = True

        return result

    def _aggregate_samples(
        self,
        qid: str,
        embedding_score: int,
        samples: List[dict],
    ) -> QuestionResult:
        """Aggregate successful LLM samples into a stable score."""
        scores = [
            int(s["score"])
            for s in samples
        ]

        mean_score = statistics.fmean(scores)

        std_score = (
            statistics.pstdev(scores)
            if len(scores) > 1
            else 0.0
        )

        representative = min(
            samples,
            key=lambda s: abs(
                s["score"] - mean_score
            ),
        )

        final_score = round(
            SIMILARITY_WEIGHT * embedding_score
            + LLM_WEIGHT * mean_score
        )

        discrepancy = abs(
            embedding_score - mean_score
        )

        uncertain = (
            std_score > UNCERTAINTY_STD_THRESHOLD
        )

        def _dedupe(items):
            seen = set()
            out = []

            for item in items:
                key = item.casefold()

                if key and key not in seen:
                    seen.add(key)
                    out.append(item)

            return out[:MAX_EVIDENCE_ITEMS]

        matched = _dedupe(
            e
            for sample in samples
            for e in sample["matched_evidence"]
        )

        missing = _dedupe(
            e
            for sample in samples
            for e in sample["missing_evidence"]
        )

        return QuestionResult(
            qid=qid,
            embedding_score=embedding_score,
            llm_scores=scores,
            llm_score_mean=round(mean_score, 1),
            llm_score_std=round(std_score, 1),
            final_score=max(
                0,
                min(100, final_score),
            ),
            reasoning=representative["reasoning"],
            matched_evidence=matched,
            missing_evidence=missing,
            discrepancy=round(
                discrepancy,
                1,
            ),
            uncertain=uncertain,
            needs_review=(
                uncertain
                or discrepancy > DISAGREEMENT_THRESHOLD
            ),
            calibration_status="ok",
        )

    def _ask_llm_batch_once(
        self,
        segments: List[QuestionSegment],
    ) -> str:
        questions_block = "\n".join(
            _QUESTION_BLOCK.format(
                qid=seg.qid,
                model_answer=seg.model_text,
                student_answer=seg.student_text,
            )
            for seg in segments
        )

        prompt = BATCH_JUDGE_PROMPT.format(
            questions_block=questions_block,
        )

        return self._complete(prompt)

    def _ask_llm_batch_with_retry(
        self,
        segments: List[QuestionSegment],
        max_retries=None,
    ):
        return self._with_retry(
            lambda: self._ask_llm_batch_once(segments),
            max_retries=max_retries,
        )

    @staticmethod
    def _normalize_qid(value):
        """
        Accepts:
            21
            "21"
            "Q21"
            "q21"
            "Question 21"

        Returns:
            "21"
        """
        if value is None:
            return None

        text = str(value).strip()

        match = re.search(
            r"(?:question\s*)?q?\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        return str(int(match.group(1)))

    @classmethod
    def _parse_batch(
        cls,
        raw_text: str,
        expected_qids: set,
    ) -> Dict[str, dict]:
        """
        Parse Mistral batch JSON robustly.

        Accepts:
          {"results": [...]}

        and dictionary-keyed output such as:
          {"21": {...}, "22": {...}}

        Also normalizes Q21 / Question 21 / 21 to "21".
        """
        cleaned = raw_text.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        ).strip()

        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            results = parsed.get("results")

            if results is None:
                results = []

                for key, value in parsed.items():
                    if isinstance(value, dict):
                        item = dict(value)
                        item.setdefault("qid", key)
                        results.append(item)

        elif isinstance(parsed, list):
            results = parsed

        else:
            raise ValueError(
                f"Unexpected batch JSON type: "
                f"{type(parsed).__name__}"
            )

        if not isinstance(results, list):
            raise ValueError(
                "Batch response does not contain "
                "a valid results array."
            )

        by_qid = {}

        for item in results:
            if not isinstance(item, dict):
                continue

            raw_qid = item.get("qid")

            if raw_qid is None:
                raw_qid = item.get("question_id")

            if raw_qid is None:
                raw_qid = item.get("question")

            qid = cls._normalize_qid(raw_qid)

            if qid is None:
                continue

            if qid not in expected_qids:
                continue

            try:
                score = int(item["score"])
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            score = max(
                0,
                min(100, score),
            )

            matched = item.get(
                "matched_evidence",
                [],
            )

            missing = item.get(
                "missing_evidence",
                [],
            )

            if not isinstance(matched, list):
                matched = [matched]

            if not isinstance(missing, list):
                missing = [missing]

            by_qid[qid] = {
                "score": score,
                "reasoning": str(
                    item.get("reasoning", "")
                ).strip(),
                "matched_evidence": [
                    str(e).strip()
                    for e in matched
                    if str(e).strip()
                ][:MAX_EVIDENCE_ITEMS],
                "missing_evidence": [
                    str(e).strip()
                    for e in missing
                    if str(e).strip()
                ][:MAX_EVIDENCE_ITEMS],
            }

        return by_qid

    def _embedding_fallback_result(
        self,
        qid: str,
        embedding_score: int,
        reason: str,
    ) -> QuestionResult:
        return QuestionResult(
            qid=qid,
            embedding_score=max(
                0,
                min(100, int(embedding_score)),
            ),
            final_score=max(
                0,
                min(100, int(embedding_score)),
            ),
            reasoning=reason,
            needs_review=True,
            calibration_status="unavailable",
        )

    def calibrate_batch(
        self,
        segments: List[QuestionSegment],
        embedding_scores: Dict[str, int],
    ) -> Dict[str, QuestionResult]:
        """
        Grade questions in small batches.

        A 429 stops remote calibration for the remainder of the run.
        A failed batch never triggers one API call per question.
        """

        if not segments:
            return {}

        expected = {
            seg.qid
            for seg in segments
        }

        # Fast path after a 429.
        if self._remote_disabled:
            return {
                qid: self._embedding_fallback_result(
                    qid=qid,
                    embedding_score=embedding_scores[qid],
                    reason=(
                        "Remote calibration disabled for this grading run; "
                        "used embedding score only. "
                        f"Reason: {self._disable_reason or 'unavailable'}"
                    ),
                )
                for qid in expected
            }

        per_qid_samples: Dict[str, list] = {
            qid: []
            for qid in expected
        }

        batches = [
            segments[i:i + self.batch_size]
            for i in range(
                0,
                len(segments),
                self.batch_size,
            )
        ]

        for batch_index, batch in enumerate(
            batches,
            start=1,
        ):
            if self._remote_disabled:
                break

            batch_qids = {
                seg.qid
                for seg in batch
            }

            print(
                f"    [CALIBRATOR] Batch "
                f"{batch_index}/{len(batches)} "
                f"({len(batch)} questions)"
            )

            for sample_index in range(
                self.n_samples
            ):
                if self._remote_disabled:
                    break

                try:
                    raw = self._ask_llm_batch_with_retry(
                        batch,
                        max_retries=self.max_retries,
                    )

                    parsed = self._parse_batch(
                        raw,
                        batch_qids,
                    )

                    print(
                        f"    [CALIBRATOR] Batch "
                        f"{batch_index} sample "
                        f"{sample_index + 1}/{self.n_samples}: "
                        f"{len(parsed)}/{len(batch_qids)} "
                        f"usable question results"
                    )

                    # Helpful diagnostic if the model returned something
                    # unexpected but parseable.
                    if not parsed:
                        print(
                            "    [CALIBRATOR] WARNING: batch response "
                            "contained no usable question results."
                        )
                        print(
                            "    [CALIBRATOR] Raw response preview:"
                        )
                        print(raw[:2000])

                    for qid, sample in parsed.items():
                        per_qid_samples[qid].append(sample)

                except RateLimitError:
                    print(
                        "    [CALIBRATOR] Batch hit HTTP 429. "
                        "Switching the remainder of this run "
                        "to deterministic fallback."
                    )
                    break

                except (
                    json.JSONDecodeError,
                    KeyError,
                    ValueError,
                    TypeError,
                ) as exc:
                    print(
                        "    [CALIBRATOR] Batch parse failure: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(
                        "    [CALIBRATOR] Raw response preview:"
                    )
                    try:
                        print(raw[:2000])
                    except Exception:
                        print(repr(raw)[:2000])

                except Exception as exc:
                    print(
                        "    [CALIBRATOR] Batch failure: "
                        f"{type(exc).__name__}: {exc}"
                    )

        results: Dict[str, QuestionResult] = {}

        for qid in expected:
            emb_score = embedding_scores[qid]
            samples = per_qid_samples[qid]

            if samples:
                result = self._aggregate_samples(
                    qid,
                    emb_score,
                    samples,
                )

                if len(samples) < self.n_samples:
                    result.calibration_status = "parse_error"
                    result.needs_review = True

                results[qid] = result
                continue

            if self._remote_disabled:
                reason = (
                    "Remote calibration was rate limited; "
                    "used embedding score only."
                )
            else:
                reason = (
                    "No valid LLM result was returned for this "
                    "question; used embedding score only."
                )

            results[qid] = (
                self._embedding_fallback_result(
                    qid=qid,
                    embedding_score=emb_score,
                    reason=reason,
                )
            )

        return results

    def calibrate_document(
        self,
        segments: List[QuestionSegment],
        embedding_scores: List[int],
    ) -> CalibrationResult:
        """
        Grade a document using batched calibration.

        With batch_size=5 and n_samples=1, 15 short answers require
        approximately 3 remote requests instead of 15.
        """

        if not segments:
            return CalibrationResult(
                final_score=0,
                embedding_score=0,
                questions=[],
                reasoning="No questions supplied.",
                needs_review=True,
                calibration_status="partial",
            )

        emb_by_qid = {
            seg.qid: int(score)
            for seg, score in zip(
                segments,
                embedding_scores,
            )
        }

        results_by_qid = self.calibrate_batch(
            segments,
            emb_by_qid,
        )

        questions = [
            results_by_qid[seg.qid]
            for seg in segments
        ]

        final_score = round(
            statistics.fmean(
                q.final_score
                for q in questions
            )
        )

        embedding_score = round(
            statistics.fmean(
                q.embedding_score
                for q in questions
            )
        )

        needs_review = any(
            q.needs_review
            for q in questions
        )

        calibration_status = (
            "ok"
            if all(
                q.calibration_status == "ok"
                for q in questions
            )
            else "partial"
        )

        if (
            len(questions) == 1
            and questions[0].qid == "full"
        ):
            reasoning = questions[0].reasoning
        else:
            reasoning = "\n".join(
                f"Q{q.qid}: {q.reasoning}"
                for q in questions
            )

        return CalibrationResult(
            final_score=max(
                0,
                min(100, final_score),
            ),
            embedding_score=max(
                0,
                min(100, embedding_score),
            ),
            questions=questions,
            reasoning=reasoning,
            needs_review=needs_review,
            calibration_status=calibration_status,
        )