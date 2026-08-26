import argparse, os, re
from datetime import datetime, timezone

from dotenv import load_dotenv
from mistralai.client import Mistral
from groq import Groq
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient

from calibrator import LLMCalibrator
from mcq_grader import grade_mcq
from mendeley_loader import MCQ_QIDS, SHORT_ANSWER_QIDS, load_answer_key
from segmenter import QuestionSegment
from rubric_scorer import score_rubric


load_dotenv(override=True)

MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
CALIBRATION_MODEL = os.getenv("CALIBRATION_MODEL", "openai/gpt-oss-120b")

if not MISTRAL_KEY:
    raise RuntimeError("MISTRAL_API_KEY is not set in .env")
if not GROQ_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in .env")

OCR_MODEL = "mistral-ocr-latest"
MONGO_URI = "mongodb://localhost:27017/aae"
VALID_QIDS = [str(i) for i in range(1, 36)]

HEADER = re.compile(
    r"(?m)^[ \t]*(\d{1,2})(?!\d)(?![ \t]*/)"
    r"[ \t]*(?:[.)\]:;-][ \t]*)?"
)


def clamp(v):
    try:
        return max(0, min(100, float(v)))
    except (TypeError, ValueError):
        return 0


def ocr_pdf(path, client):
    with open(path, "rb") as f:
        upload = client.files.upload(
            file={
                "file_name": os.path.basename(path),
                "content": f,
            },
            purpose="ocr",
        )

    url = client.files.get_signed_url(file_id=upload.id).url
    response = client.ocr.process(
        model=OCR_MODEL,
        document={
            "type": "document_url",
            "document_url": url,
        },
    )
    return "\n".join(p.markdown for p in response.pages)


def similarity(embedder, a, b):
    if not a.strip() or not b.strip():
        return 0

    x = embedder.encode(
        a, convert_to_tensor=True, normalize_embeddings=True
    )
    y = embedder.encode(
        b, convert_to_tensor=True, normalize_embeddings=True
    )

    return round(clamp(util.cos_sim(x, y).item() * 100))


def split_mendeley_text(text):
    matches = [
        (m.group(1), m)
        for m in HEADER.finditer(text)
        if m.group(1) in VALID_QIDS
    ]

    segments, duplicates, seen = {}, [], set()

    for i, (qid, match) in enumerate(matches):
        end = (
            matches[i + 1][1].start()
            if i + 1 < len(matches)
            else len(text)
        )
        body = text[match.end():end].strip()

        if not body:
            continue

        if qid in seen:
            duplicates.append((qid, body))
        else:
            seen.add(qid)
            segments[qid] = body

    return segments, duplicates


def _best_assignment(embedder, text, missing, key):
    if not text.strip() or not missing:
        return None, 0

    chunk = embedder.encode(
        text, convert_to_tensor=True, normalize_embeddings=True
    )

    best_q, best_score = None, -1

    for qid in missing:
        answer = key[qid].model_answer.strip()
        if not answer:
            continue

        ref = embedder.encode(
            answer, convert_to_tensor=True, normalize_embeddings=True
        )
        score = util.cos_sim(chunk, ref).item() * 100

        if score > best_score:
            best_q, best_score = qid, score

    return best_q, round(clamp(best_score))


def recover_missing(segments, duplicates, key, embedder, threshold=45):
    missing = [
        q for q in SHORT_ANSWER_QIDS
        if q in key and q not in segments
    ]

    if not missing or not duplicates:
        return segments, []

    remaining = set(missing)
    recovered = []

    for src, text in sorted(
        duplicates, key=lambda x: len(x[1]), reverse=True
    ):
        if not remaining:
            break

        qid, score = _best_assignment(
            embedder, text, sorted(remaining, key=int), key
        )

        if qid is None or score < threshold:
            continue

        segments[qid] = text
        remaining.remove(qid)

        recovered.append({
            "from_qid": src,
            "to_qid": qid,
            "similarity": score,
        })

    return segments, recovered


def build_short_data(segments, key, embedder):
    qs, embeddings, rubrics = [], {}, {}

    for qid in SHORT_ANSWER_QIDS:
        if qid not in key:
            continue

        answer = segments.get(qid, "").strip() or "[No answer provided]"
        segment = QuestionSegment(
            qid=qid,
            model_text=key[qid].model_answer,
            student_text=answer,
        )

        qs.append(segment)
        embeddings[qid] = similarity(
            embedder,
            key[qid].model_answer,
            answer,
        )
        rubrics[qid] = score_rubric(
            key[qid].keypoints,
            answer,
        )

    return qs, embeddings, rubrics


def grade_student(student_id, pdf_path, key, mistral, embedder, calibrator):
    print(f"OCR provider : Mistral / {OCR_MODEL}")
    print(f"Calibration provider : Groq / {calibrator.model}")

    text = ocr_pdf(pdf_path, mistral)
    segments, duplicates = split_mendeley_text(text)

    print(f"OCR segmentation before recovery: {len(segments)}/35")
    if duplicates:
        print(f"Duplicate/misnumbered chunks: {len(duplicates)}")

    segments, recovered = recover_missing(
        segments, duplicates, key, embedder
    )

    for item in recovered:
        print(
            f"Recovered OCR chunk {item['from_qid']} -> "
            f"Q{item['to_qid']} "
            f"(semantic similarity {item['similarity']})"
        )

    missing = [
        q for q in VALID_QIDS
        if q not in segments
    ]

    print(f"OCR segmentation after recovery: {len(segments)}/35")
    if missing:
        print(
            "  Still missing: "
            + ", ".join(f"Q{q}" for q in missing)
        )

    mcqs = [
        grade_mcq(
            qid,
            segments.get(qid, ""),
            key[qid],
        )
        for qid in MCQ_QIDS
        if qid in key
    ]

    qs, embeddings, rubrics = build_short_data(
        segments, key, embedder
    )

    print(f"Short-answer questions: {len(qs)}")
    print(
        f"Starting Groq calibration: "
        f"{len(qs)} questions, "
        f"n_samples={calibrator.n_samples}, "
        f"batch_size={calibrator.batch_size}"
    )

    llm = calibrator.calibrate_batch(
        qs,
        embeddings,
    )

    short = []

    for q in qs:
        llm_result = llm[q.qid]
        rubric = rubrics[q.qid]
        emb = embeddings[q.qid]

        if llm_result.llm_score_mean is None:
            final = round((emb + rubric.score) / 2)
        else:
            final = round(
                0.33 * emb
                + 0.34 * llm_result.llm_score_mean
                + 0.33 * rubric.score
            )

        review = (
            llm_result.needs_review
            or llm_result.calibration_status != "ok"
        )

        short.append({
            "qid": q.qid,
            "student_answer": q.student_text,
            "embedding_score": emb,
            "llm_result": llm_result,
            "rubric_result": rubric,
            "final_score": clamp(final),
            "needs_review": review,
        })

    mcq_marks = sum(r.score / 100 for r in mcqs)
    short_marks = sum(
        r["final_score"] / 100 * 2
        for r in short
    )

    return {
        "student_id": student_id,
        "total_marks": round(
            mcq_marks + short_marks, 2
        ),
        "max_marks": 50,
        "mcq_marks": round(mcq_marks, 2),
        "short_answer_marks": round(short_marks, 2),
        "needs_review": (
            bool(missing)
            or any(r.needs_review for r in mcqs)
            or any(r["needs_review"] for r in short)
        ),
        "mcq_results": mcqs,
        "short_results": short,
        "missing_qids": missing,
        "recovered": recovered,
        "created_at": datetime.now(timezone.utc),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Grade the Mendeley exam dataset."
    )
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--student",
        nargs="+",
        help="Specific student IDs, e.g. Student_10 Student_11",
    )
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--requests-per-second", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    key = load_answer_key(
        os.path.join(args.dataset_dir, "answerkey.txt")
    )
    pdf_dir = os.path.join(
        args.dataset_dir,
        "Student_Pdf",
    )

    mistral = Mistral(api_key=MISTRAL_KEY)
    groq = Groq(api_key=GROQ_KEY)

    calibrator = LLMCalibrator(
        groq,
        model=CALIBRATION_MODEL,
        n_samples=args.n_samples,
        requests_per_second=args.requests_per_second,
        batch_size=args.batch_size,
    )

    print("Calibration provider: GROQ")
    print(f"Calibration model : {CALIBRATION_MODEL}")

    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    files = sorted(
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    )

    if args.student:
        wanted = set(args.student)
        files = [
            f for f in files
            if os.path.splitext(f)[0] in wanted
        ]

    elif args.limit:
        files = files[:args.limit]

    mongo = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
    )
    mongo.admin.command("ping")
    collection = mongo["aae"]["results"]

    for filename in files:
        student_id = os.path.splitext(
            filename
        )[0]

        print(f"\nGrading {student_id}...")
        calibrator.reset()

        result = grade_student(
            student_id,
            os.path.join(pdf_dir, filename),
            key,
            mistral,
            embedder,
            calibrator,
        )

        collection.insert_one({
            "student_id": student_id,
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
                    "qid": r["qid"],
                    "student_answer": r["student_answer"],
                    "embedding_score": r["embedding_score"],
                    "llm_scores": r["llm_result"].llm_scores,
                    "llm_score_mean": r["llm_result"].llm_score_mean,
                    "llm_score_std": r["llm_result"].llm_score_std,
                    "calibration_status": r["llm_result"].calibration_status,
                    "reasoning": r["llm_result"].reasoning,
                    "matched_evidence": r["llm_result"].matched_evidence,
                    "missing_evidence": r["llm_result"].missing_evidence,
                    "discrepancy": r["llm_result"].discrepancy,
                    "rubric_score": r["rubric_result"].score,
                    "rubric_matched": r["rubric_result"].matched_keypoints,
                    "rubric_missed": r["rubric_result"].missed_keypoints,
                    "final_score": r["final_score"],
                    "needs_review": r["needs_review"],
                }
                for r in result["short_results"]
            ],
            "created_at": result["created_at"],
        })

        print(
            f"Total: {result['total_marks']}/50"
            f"{'REVIEW' if result['needs_review'] else ''}"
        )

        for r in result["mcq_results"]:
            mark = (
                "Correct!" if r.is_correct
                else "?" if r.needs_review
                else "Wrong!"
            )
            print(
                f" Q{r.qid} [MCQ] {mark} "
                f"student={r.student_option!r} "
                f"correct={r.correct_option!r}"
            )

        for r in result["short_results"]:
            l, rub = r["llm_result"], r["rubric_result"]
            std = (
                f" ±{l.llm_score_std}"
                if l.llm_score_std is not None
                else " n/a"
            )

            print(
                f"    Q{r['qid']} [SHORT] "
                f"{r['final_score']}/100 "
                f"(emb {r['embedding_score']}, "
                f"llm {l.llm_score_mean}{std}, "
                f"rubric {rub.score})"
                f"{'  REVIEW' if r['needs_review'] else ''}"
            )

            if l.reasoning:
                print(f"      Reason: {l.reasoning}")
            if l.matched_evidence:
                print(
                    "      Matched: "
                    + "; ".join(l.matched_evidence)
                )
            if l.missing_evidence:
                print(
                    "      Missing: "
                    + "; ".join(l.missing_evidence)
                )


if __name__ == "__main__":
    main()