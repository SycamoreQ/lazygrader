from pathlib import Path
import argparse
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from mistralai.client import Mistral
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient

from calibrator import LLMCalibrator
from mcq_grader import grade_mcq
from mendeley_loader import MCQ_QIDS, SHORT_ANSWER_QIDS, load_answer_key
from segmenter import QuestionSegment
from short_answer_grader import combine_short_answer_signals


load_dotenv(override=True)

API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is not set in .env")

OCR_MODEL = "mistral-ocr-latest"
CALIBRATION_MODEL = "mistral-large-latest"

MENDELEY_DIR = "data"
MONGO_URI = "mongodb://localhost:27017/aae"

VALID_QIDS = [str(i) for i in range(1, 36)]

# A line is treated as a question header only when the number is known.
# Allows:
#   28.
#   28)
#   28:
#   28 -
#   28.Continuous
#   28 Continuous
# It deliberately does not treat "1/total 3" as a header.
_NUMERIC_HEADER = re.compile(
    r"(?m)^[ \t]*(\d{1,2})(?!\d)"
    r"(?![ \t]*/)"                       # reject page markers such as 1/3
    r"[ \t]*(?:[.)\]:;-][ \t]*)?"        # optional punctuation
)


def clamp_score(value, lo=0.0, hi=100.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, value))


def extract_text_from_pdf(pdf_path, client):
    with open(pdf_path, "rb") as f:
        upload = client.files.upload(
            file={
                "file_name": os.path.basename(pdf_path),
                "content": f,
            },
            purpose="ocr",
        )

    signed_url = client.files.get_signed_url(file_id=upload.id)

    response = client.ocr.process(
        model=OCR_MODEL,
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
    )

    return "\n".join(page.markdown for page in response.pages)


def embed_similarity_score(embedder, text_a, text_b):
    if not text_a.strip() or not text_b.strip():
        return 0

    a = embedder.encode(text_a, convert_to_tensor=True)
    b = embedder.encode(text_b, convert_to_tensor=True)

    cosine = util.cos_sim(a, b).item()
    # Use [0,100] for grading rather than allowing negative values.
    return round(clamp_score(cosine * 100) or 0)


def split_mendeley_text(text, valid_qids):
    """
    OCR-tolerant numeric segmentation.

    Returns:
        segments: first chunk for each qid
        duplicates: later chunks carrying an already-seen qid
    """
    valid = set(valid_qids)
    raw_matches = []

    for m in _NUMERIC_HEADER.finditer(text):
        qid = m.group(1)
        if qid not in valid:
            continue

        # Reject obvious page/date/table artifacts.
        line_end = text.find("\n", m.end())
        if line_end == -1:
            line_end = len(text)
        header_line_tail = text[m.end():line_end].strip()

        if header_line_tail.startswith("/"):
            continue

        raw_matches.append((qid, m))

    segments = {}
    duplicates = []
    seen = set()

    for i, (qid, match) in enumerate(raw_matches):
        start = match.end()
        end = raw_matches[i + 1][1].start() if i + 1 < len(raw_matches) else len(text)
        body = text[start:end].strip()

        if not body:
            continue

        if qid not in seen:
            seen.add(qid)
            segments[qid] = body
        else:
            duplicates.append((qid, body))

    return segments, duplicates


def _best_assignment(embedder, chunk_text, missing_qids, answer_key):
    """
    Pick the missing qid whose model answer is semantically closest to a
    duplicate/misnumbered OCR chunk.
    """
    if not chunk_text.strip() or not missing_qids:
        return None, 0

    chunk_emb = embedder.encode(chunk_text, convert_to_tensor=True)

    candidates = []
    for qid in missing_qids:
        model_answer = answer_key[qid].model_answer.strip()
        if not model_answer:
            continue

        answer_emb = embedder.encode(model_answer, convert_to_tensor=True)
        sim = util.cos_sim(chunk_emb, answer_emb).item()
        candidates.append((sim, qid))

    if not candidates:
        return None, 0

    sim, qid = max(candidates)
    return qid, round(clamp_score(sim * 100) or 0)


def reconcile_missing_questions(
    segments,
    duplicates,
    answer_key,
    embedder,
    min_recovery_similarity=45,
):
    """
    Recover missing short-answer qids from duplicate/misnumbered chunks.

    This is intentionally conservative:
    - only Q21-Q35 are candidates for recovery
    - each missing qid can be assigned once
    - only recover when semantic similarity reaches the threshold
    """
    missing = [
        qid
        for qid in SHORT_ANSWER_QIDS
        if qid in answer_key and qid not in segments
    ]

    if not missing or not duplicates:
        return segments, []

    recovered = []
    remaining_missing = set(missing)

    # Longer chunks first: they generally carry more answer evidence.
    candidates = sorted(
        duplicates,
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for original_qid, chunk in candidates:
        if not remaining_missing:
            break

        target_qid, similarity = _best_assignment(
            embedder,
            chunk,
            sorted(remaining_missing, key=int),
            answer_key,
        )

        if target_qid is None or similarity < min_recovery_similarity:
            continue

        segments[target_qid] = chunk
        remaining_missing.remove(target_qid)

        recovered.append(
            {
                "from_qid": original_qid,
                "to_qid": target_qid,
                "similarity": similarity,
            }
        )

    return segments, recovered


def _safe_llm_status(result):
    llm_result = getattr(result, "llm_result", None)
    mean = getattr(llm_result, "llm_score_mean", None)
    std = getattr(llm_result, "llm_score_std", None)

    # Different versions of calibrator may expose one of these.
    status = getattr(llm_result, "calibration_status", None)
    if status is None:
        status = getattr(result, "calibration_status", None)

    reasoning = getattr(result, "reasoning", "") or ""

    return mean, std, status, reasoning


def _fallback_score(result):
    """
    Explicit fallback ONLY when LLM calibration is unavailable.
    This keeps the score bounded and marks the question for review.
    """
    emb = clamp_score(getattr(result, "embedding_score", 0)) or 0
    rubric_obj = getattr(result, "rubric_result", None)
    rubric = clamp_score(getattr(rubric_obj, "score", 0)) or 0

    # Do not pretend a failed LLM call has equal evidence to a successful
    # calibration. Here we use the two deterministic signals only.
    result.final_score = round((emb + rubric) / 2)
    result.needs_review = True
    return result


def grade_student(
    student_id,
    pdf_path,
    answer_key,
    mistral_client,
    embedder,
    calibrator,
):
    student_text = extract_text_from_pdf(pdf_path, mistral_client)

    segments, duplicates = split_mendeley_text(
        student_text,
        VALID_QIDS,
    )

    original_missing = [
        qid for qid in VALID_QIDS if qid not in segments
    ]

    print(
        f"  OCR segmentation before recovery: "
        f"{len(segments)}/{len(VALID_QIDS)}"
    )

    if duplicates:
        print(f"  Duplicate/misnumbered chunks: {len(duplicates)}")

    segments, recovered = reconcile_missing_questions(
        segments,
        duplicates,
        answer_key,
        embedder,
    )

    if recovered:
        for item in recovered:
            print(
                f"  Recovered OCR chunk {item['from_qid']} -> "
                f"Q{item['to_qid']} "
                f"(semantic similarity {item['similarity']})"
            )

    missing = [
        qid for qid in VALID_QIDS if qid not in segments
    ]

    print(
        f"  OCR segmentation after recovery: "
        f"{len(segments)}/{len(VALID_QIDS)}"
    )

    if missing:
        print(
            "  Still missing: "
            + ", ".join("Q" + q for q in missing)
        )

    # -------------------------
    # MCQ
    # -------------------------
    mcq_results = [
        grade_mcq(
            qid,
            segments.get(qid, ""),
            answer_key[qid],
        )
        for qid in MCQ_QIDS
        if qid in answer_key
    ]

    # -------------------------
    # SHORT ANSWERS
    # -------------------------
    # Build every question's segment + embedding score first (both are
    # local -- no API calls), then grade the whole set in ONE calibrator
    # call (per self-consistency sample) instead of one call per question.
    # See calibrator.calibrate_batch for why this is the actual fix for
    # rate-limit exhaustion, not just a slower version of the old loop.
    short_segments = []
    short_key_entries = {}
    short_embedding_scores = {}

    for qid in SHORT_ANSWER_QIDS:
        if qid not in answer_key:
            continue

        key_entry = answer_key[qid]

        student_body = (
            segments.get(qid, "").strip()
            or "[No answer provided]"
        )

        segment = QuestionSegment(
            qid=qid,
            model_text=key_entry.model_answer,
            student_text=student_body,
        )

        short_segments.append(segment)
        short_key_entries[qid] = key_entry
        short_embedding_scores[qid] = embed_similarity_score(
            embedder,
            key_entry.model_answer,
            student_body,
        )

    try:
        llm_results = calibrator.calibrate_batch(short_segments, short_embedding_scores)
    except Exception as exc:
        print(f"    Batch LLM calibration exception: {type(exc).__name__}: {exc}")
        llm_results = {}

    short_results = []

    for segment in short_segments:
        qid = segment.qid
        key_entry = short_key_entries[qid]
        embedding_score = short_embedding_scores[qid]
        llm_result = llm_results.get(qid)

        if llm_result is None:
            # The batch call itself raised before producing any per-
            # question result -- there is no legitimate LLM score here.
            print(f"    Q{qid} has no grading result -> REVIEW")
            continue

        result = combine_short_answer_signals(
            segment,
            key_entry,
            embedding_score,
            llm_result,
        )

        llm_mean, llm_std, llm_status, reasoning = _safe_llm_status(result)

        if llm_mean is None:
            print(
                f"    Q{qid} LLM unavailable"
                + (f" (status={llm_status})" if llm_status else "")
            )
            if reasoning:
                print(f"         reason: {reasoning[:240]}")

            result = _fallback_score(result)
        else:
            result.final_score = round(
                clamp_score(getattr(result, "final_score", 0)) or 0
            )

        short_results.append(result)

    mcq_marks = sum(r.score / 100 for r in mcq_results)
    short_marks = sum(r.final_score / 100 * 2 for r in short_results)

    # Important: max_marks stays 50 only when all 15 short answers produced
    # result records. A missing result is a pipeline failure and is flagged.
    max_marks = 20 + 15 * 2

    pipeline_review = (
        len(short_results) != len(SHORT_ANSWER_QIDS)
        or bool(missing)
    )

    return {
        "student_id": student_id,
        "total_marks": round(mcq_marks + short_marks, 2),
        "max_marks": max_marks,
        "mcq_marks": round(mcq_marks, 2),
        "short_answer_marks": round(short_marks, 2),
        "needs_review": (
            pipeline_review
            or any(r.needs_review for r in mcq_results)
            or any(r.needs_review for r in short_results)
        ),
        "mcq_results": mcq_results,
        "short_results": short_results,
        "missing_qids": missing,
        "recovered": recovered,
        "created_at": datetime.now(timezone.utc),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Grade the Mendeley exam dataset."
    )

    parser.add_argument(
        "--dataset-dir",
        default=MENDELEY_DIR,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help=(
            "LLM self-consistency samples. Start with 1 while debugging "
            "API/rate-limit behaviour."
        ),
    )

    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=1.0,
        help=(
            "Max Mistral chat-completion calls/second for the LLM judge. "
            "Defaults to 1.0, matching Mistral's Free/Experiment tier "
            "(https://admin.mistral.ai/plateforme/limits shows your "
            "workspace's actual tier -- raise this if yours allows more)."
        ),
    )

    args = parser.parse_args()

    answer_key_path = os.path.join(
        args.dataset_dir,
        "answerkey.txt",
    )
    student_pdf_dir = os.path.join(
        args.dataset_dir,
        "Student_Pdf",
    )

    mistral_client = Mistral(api_key=API_KEY)

    calibrator = LLMCalibrator(
        mistral_client,
        model=CALIBRATION_MODEL,
        n_samples=args.n_samples,
        requests_per_second=args.requests_per_second,
    )

    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    answer_key = load_answer_key(answer_key_path)

    if not answer_key:
        raise RuntimeError(
            f"No answer key entries parsed from {answer_key_path}"
        )

    mongo_client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
    )
    mongo_client.admin.command("ping")
    collection = mongo_client["aae"]["mendeley_results"]

    filenames = sorted(
        f
        for f in os.listdir(student_pdf_dir)
        if f.lower().endswith(".pdf")
    )

    if args.limit:
        filenames = filenames[:args.limit]

    for filename in filenames:
        student_id = os.path.splitext(filename)[0]
        pdf_path = os.path.join(student_pdf_dir, filename)

        print(f"\nGrading {student_id}...")

        result = grade_student(
            student_id,
            pdf_path,
            answer_key,
            mistral_client,
            embedder,
            calibrator,
        )

        doc = {
            "student_id": result["student_id"],
            "total_marks": result["total_marks"],
            "max_marks": result["max_marks"],
            "mcq_marks": result["mcq_marks"],
            "short_answer_marks": result["short_answer_marks"],
            "needs_review": result["needs_review"],
            "missing_qids": result["missing_qids"],
            "recovered": result["recovered"],
            "mcq_breakdown": [
                {
                    "qid": r.qid,
                    "correct_option": r.correct_option,
                    "student_option": r.student_option,
                    "is_correct": r.is_correct,
                    "needs_review": r.needs_review,
                }
                for r in result["mcq_results"]
            ],
            "short_answer_breakdown": [
                {
                    "qid": r.qid,
                    "embedding_score": r.embedding_score,
                    "llm_score_mean": getattr(
                        r.llm_result,
                        "llm_score_mean",
                        None,
                    ),
                    "llm_score_std": getattr(
                        r.llm_result,
                        "llm_score_std",
                        None,
                    ),
                    "rubric_score": getattr(
                        r.rubric_result,
                        "score",
                        0,
                    ),
                    "rubric_matched": getattr(
                        r.rubric_result,
                        "matched_keypoints",
                        [],
                    ),
                    "rubric_missed": getattr(
                        r.rubric_result,
                        "missed_keypoints",
                        [],
                    ),
                    "final_score": r.final_score,
                    "reasoning": r.reasoning,
                    "needs_review": r.needs_review,
                }
                for r in result["short_results"]
            ],
            "created_at": result["created_at"],
        }

        collection.insert_one(doc)

        print(
            f"  Total: {result['total_marks']}/{result['max_marks']}"
            f"{'  REVIEW' if result['needs_review'] else ''}"
        )

        for r in result["mcq_results"]:
            mark = (
                "✓"
                if r.is_correct
                else ("?" if r.needs_review else "✗")
            )
            print(
                f"    Q{r.qid} [MCQ] {mark} "
                f"student={r.student_option!r} "
                f"correct={r.correct_option!r}"
            )

        for r in result["short_results"]:
            llm_mean = getattr(
                r.llm_result,
                "llm_score_mean",
                None,
            )
            llm_std = getattr(
                r.llm_result,
                "llm_score_std",
                None,
            )
            rubric = getattr(
                r.rubric_result,
                "score",
                0,
            )

            flag = "  REVIEW" if r.needs_review else ""

            print(
                f"    Q{r.qid} [SHORT] "
                f"{r.final_score}/100 "
                f"(emb {r.embedding_score}, "
                f"llm {llm_mean}, "
                f"rubric {rubric})"
                f"{flag}"
            )


if __name__ == "__main__":
    main()