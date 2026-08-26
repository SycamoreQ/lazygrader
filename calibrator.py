import json, re, statistics, time
from dataclasses import dataclass, field
from groq import Groq
from segmenter import QuestionSegment

MODEL = "openai/gpt-oss-120b"
N_SAMPLES, TEMP, RPS, BATCH = 1, 0.2, 0.5, 5
DISAGREE, STD_LIMIT, MAX_EVIDENCE = 20, 12, 5

PROMPT = """You are an impartial exam grader.

Grade every question independently.

SCORING:
90-100 fully correct
75-89 mostly correct
60-74 substantially correct
40-59 partially correct
20-39 limited understanding
1-19 very little relevant content
0 no relevant answer

IMPORTANT:
- score MUST be an integer 0-100
- NEVER use a 0-5 or 0-10 scale
- NEVER use letter grades
- ignore minor OCR spelling/transcription errors
- explain the score briefly
- identify matched and missing evidence
- return ONLY valid JSON

{questions}

Return:
{{"results":[{{"qid":"21","score":75,"reasoning":"...",
"matched_evidence":["..."],"missing_evidence":["..."]}}]}}
"""

BLOCK = """--- Question {qid} ---
MODEL ANSWER:
{model_answer}

STUDENT ANSWER:
{student_answer}
"""


class RateLimitError(Exception):
    pass


@dataclass
class QuestionResult:
    qid: str
    embedding_score: int
    llm_scores: list = field(default_factory=list)
    llm_score_mean: float | None = None
    llm_score_std: float | None = None
    final_score: int = 0
    reasoning: str = ""
    matched_evidence: list = field(default_factory=list)
    missing_evidence: list = field(default_factory=list)
    discrepancy: float | None = None
    uncertain: bool = False
    needs_review: bool = False
    calibration_status: str = "ok"


class LLMCalibrator:
    def __init__(self, client: Groq, model=MODEL, n_samples=N_SAMPLES,
                 requests_per_second=RPS, max_retries=0, batch_size=BATCH):
        self.client, self.model = client, model
        self.n_samples = max(1, int(n_samples))
        self.max_retries = max(0, int(max_retries))
        self.batch_size = max(1, int(batch_size))
        self.interval = 1 / requests_per_second if requests_per_second > 0 else 0
        self.last_call = 0.0
        self.disabled = False
        self.reason = None

    def reset(self):
        self.disabled = False
        self.reason = None
        self.last_call = 0.0

    def _wait(self):
        if self.interval:
            left = self.interval - (time.monotonic() - self.last_call)
            if left > 0:
                time.sleep(left)
            self.last_call = time.monotonic()

    @staticmethod
    def _rate_limited(e):
        s = str(e).lower()
        return any(x in s for x in
                   ("429", "rate limit", "rate_limited", "too many requests"))

    @staticmethod
    def _qid(x):
        m = re.search(r"(?:question\s*)?q?\s*(\d+)", str(x), re.I) if x is not None else None
        return str(int(m.group(1))) if m else None

    def _disable(self, reason):
        self.disabled, self.reason = True, reason
        print(f"[CALIBRATOR] GROQ disabled for this student: {reason}")

    def _call(self, prompt):
        if self.disabled:
            raise RateLimitError(self.reason or "Groq disabled")
        self._wait()
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an exam grader. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMP,
                response_format={"type": "json_object"},
            )
            return r.choices[0].message.content
        except Exception as e:
            if self._rate_limited(e):
                self._disable("Groq returned HTTP 429 rate_limited.")
                raise RateLimitError(self.reason) from e
            raise

    def _request(self, prompt):
        for attempt in range(self.max_retries + 1):
            try:
                return self._call(prompt)
            except RateLimitError:
                raise
            except Exception:
                if attempt >= self.max_retries:
                    raise
                delay = 2 ** attempt
                print(f"[CALIBRATOR] transient Groq error; retrying in {delay}s")
                time.sleep(delay)

    @staticmethod
    def _clean(raw):
        return re.sub(r"\s*```$", "", re.sub(
            r"^```(?:json)?\s*", "", str(raw).strip(), flags=re.I
        )).strip()

    @classmethod
    def _parse_batch(cls, raw, expected):
        data = json.loads(cls._clean(raw))
        if not isinstance(data.get("results"), list):
            raise ValueError("Groq response has no results array")
        out = {}
        for x in data["results"]:
            if not isinstance(x, dict):
                continue
            qid = cls._qid(x.get("qid"))
            if qid not in expected:
                continue
            try:
                score = max(0, min(100, int(x["score"])))
            except (KeyError, TypeError, ValueError):
                continue
            norm = lambda v: [str(i).strip() for i in (v if isinstance(v, list) else [v]) if str(i).strip()][:MAX_EVIDENCE]
            out[qid] = {
                "score": score,
                "reasoning": str(x.get("reasoning", "")).strip(),
                "matched_evidence": norm(x.get("matched_evidence", [])),
                "missing_evidence": norm(x.get("missing_evidence", [])),
            }
        return out

    @staticmethod
    def _fallback(qid, emb, reason):
        return QuestionResult(
            qid=qid, embedding_score=emb, final_score=emb,
            reasoning=reason, needs_review=True,
            calibration_status="unavailable"
        )

    def _aggregate(self, qid, emb, samples):
        scores = [s["score"] for s in samples]
        mean = statistics.fmean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        rep = min(samples, key=lambda s: abs(s["score"] - mean))
        matched = list(dict.fromkeys(
            x for s in samples for x in s["matched_evidence"] if x
        ))[:MAX_EVIDENCE]
        missing = list(dict.fromkeys(
            x for s in samples for x in s["missing_evidence"] if x
        ))[:MAX_EVIDENCE]
        discrepancy = abs(emb - mean)

        return QuestionResult(
            qid=qid,
            embedding_score=emb,
            llm_scores=scores,
            llm_score_mean=round(mean, 1),
            llm_score_std=round(std, 1),
            final_score=round(0.5 * emb + 0.5 * mean),
            reasoning=rep["reasoning"],
            matched_evidence=matched,
            missing_evidence=missing,
            discrepancy=round(discrepancy, 1),
            uncertain=std > STD_LIMIT,
            needs_review=std > STD_LIMIT or discrepancy > DISAGREE,
        )

    def calibrate_batch(self, segments: list[QuestionSegment], embeddings: dict):
        expected = {s.qid for s in segments}
        samples = {q: [] for q in expected}

        if not segments:
            return {}

        if self.disabled:
            return {q: self._fallback(q, embeddings[q], self.reason or "Groq unavailable.")
                    for q in expected}

        batches = [segments[i:i + self.batch_size]
                   for i in range(0, len(segments), self.batch_size)]

        for bi, batch in enumerate(batches, 1):
            if self.disabled:
                break

            print(f"[CALIBRATOR] GROQ batch {bi}/{len(batches)} ({len(batch)} questions)")
            prompt = PROMPT.format(questions="\n".join(
                BLOCK.format(
                    qid=s.qid,
                    model_answer=s.model_text,
                    student_answer=s.student_text
                ) for s in batch
            ))

            for si in range(self.n_samples):
                if self.disabled:
                    break
                try:
                    parsed = self._parse_batch(
                        self._request(prompt),
                        {s.qid for s in batch}
                    )
                    print(
                        f"[CALIBRATOR] batch {bi} sample {si + 1}/{self.n_samples}: "
                        f"{len(parsed)}/{len(batch)} usable results"
                    )
                    for qid, value in parsed.items():
                        samples[qid].append(value)
                except RateLimitError:
                    print("[CALIBRATOR] Switching remaining questions to fallback.")
                    break
                except Exception as e:
                    print(f"[CALIBRATOR] Groq batch error: {type(e).__name__}: {e}")

        result = {}
        for qid in expected:
            if samples[qid]:
                r = self._aggregate(qid, embeddings[qid], samples[qid])
                if len(samples[qid]) < self.n_samples:
                    r.calibration_status = "parse_error"
                    r.needs_review = True
                result[qid] = r
            else:
                result[qid] = self._fallback(
                    qid, embeddings[qid],
                    self.reason or "No valid Groq result."
                )
        return result

    def calibrate_question(self, segment, embedding_score):
        """Compatibility path for code that still calls this method."""
        return self.calibrate_batch([segment], {segment.qid: embedding_score})[segment.qid]

    def calibrate_document(self, segments, embedding_scores):
        results = self.calibrate_batch(
            segments,
            {s.qid: v for s, v in zip(segments, embedding_scores)}
        )
        qs = [results[s.qid] for s in segments]
        return type("CalibrationResult", (), {
            "final_score": round(statistics.fmean(q.final_score for q in qs)) if qs else 0,
            "embedding_score": round(statistics.fmean(q.embedding_score for q in qs)) if qs else 0,
            "questions": qs,
            "reasoning": "\n".join(f"Q{q.qid}: {q.reasoning}" for q in qs),
            "needs_review": any(q.needs_review for q in qs),
            "calibration_status": "ok" if all(q.calibration_status == "ok" for q in qs) else "partial",
        })()


if __name__ == "__main__":
    print("calibrator.py is a library module.")