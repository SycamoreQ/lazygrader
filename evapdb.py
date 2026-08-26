import os
from datetime import datetime, timezone
from urllib.parse import quote_plus

from mistralai.client import Mistral
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient

from calibrator import LLMCalibrator
from segmenter import segment_answers


# =========================
# CONFIG
# =========================
# Prefer environment variables (see .env.example); fall back to the old
# placeholders so the script still runs if someone hasn't set up a .env yet.
# Never commit real keys here.
API_KEY = os.getenv("MISTRAL_API_KEY")

MODEL_ANSWER_PDF = "model ans.pdf"
STUDENT_ANSWER_PDF = "kanihw.pdf"

OCR_MODEL = "mistral-ocr-latest"
CALIBRATION_MODEL = "mistral-large-latest"

MONGO_USERNAME = os.getenv("MONGO_USER", "your_username")
MONGO_PASSWORD = quote_plus(os.getenv("MONGO_PASS", "your_password"))
MONGO_URI = (
    f"mongodb://localhost:27017/aae"
)


# =========================
# DATABASE
# =========================
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["aae"]
collection = db["results"]


# =========================
# OCR
# =========================
def extract_text_from_pdf(pdf_path, client):
    upload = client.files.upload(
        file={
            "file_name": os.path.basename(pdf_path),
            "content": open(pdf_path, "rb")
        },
        purpose="ocr"
    )

    signed_url = client.files.get_signed_url(file_id=upload.id)

    ocr_response = client.ocr.process(
        model=OCR_MODEL,
        document={
            "type": "document_url",
            "document_url": signed_url.url
        }
    )

    return "\n".join(page.markdown for page in ocr_response.pages)


# =========================
# SIMILARITY
# =========================
def embed_similarity_score(embedder, text_a, text_b):
    """0-100 SBERT cosine-similarity score between two text segments."""
    if not text_a.strip() or not text_b.strip():
        return 0
    emb1 = embedder.encode(text_a, convert_to_tensor=True)
    emb2 = embedder.encode(text_b, convert_to_tensor=True)
    return round(util.cos_sim(emb1, emb2).item() * 100)


# =========================
# EVALUATION
# =========================
def grade_for_score(score):
    """Map a final 0-100 score to (status, grade, template feedback)."""
    if score < 45:
        return "FAIL", "F", (
            "Your answer does not meet the minimum requirement. "
            "Several key ideas are missing or unclear."
        )

    if score >= 90:
        return "PASS", "O", "Outstanding answer with excellent clarity."
    elif score >= 80:
        return "PASS", "A", "Very good answer. Minor improvements can make it perfect."
    elif score >= 70:
        return "PASS", "B+", "Good understanding shown."
    elif score >= 60:
        return "PASS", "B", "Basic understanding is present."
    else:
        return "PASS", "C", "Average answer. Important details are missing."


# =========================
# MAIN
# =========================
def main():
    student_reg_no = input("Enter Student Register Number: ").strip()

    mistral_client = Mistral(api_key=API_KEY)
    calibrator = LLMCalibrator(mistral_client, model=CALIBRATION_MODEL)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("📄 Extracting Model Answer...")
    model_text = extract_text_from_pdf(MODEL_ANSWER_PDF, mistral_client)

    print("📝 Extracting Student Answer...")
    student_text = extract_text_from_pdf(STUDENT_ANSWER_PDF, mistral_client)

    print("✂️  Segmenting answer sheet by question...")
    segments = segment_answers(model_text, student_text)
    print(f"   Found {len(segments)} question(s): {', '.join('Q' + s.qid for s in segments)}")

    print("📊 Calculating per-question similarity...")
    embedding_scores = [embed_similarity_score(embedder, s.model_text, s.student_text) for s in segments]

    print(f"🧭 Calibrating with self-consistency judge ({calibrator.n_samples} samples/question)...")
    result = calibrator.calibrate_document(segments, embedding_scores)

    status, grade, template_feedback = grade_for_score(result.final_score)
    feedback = result.reasoning if result.calibration_status != "unavailable" else template_feedback

    # Store in MongoDB
    doc = {
        "student_reg_no": student_reg_no,
        "score": result.final_score,
        "grade": grade,
        "status": status,
        "feedback": feedback,
        "embedding_score": result.embedding_score,
        "needs_review": result.needs_review,
        "calibration_status": result.calibration_status,
        "questions": [
            {
                "qid": q.qid,
                "embedding_score": q.embedding_score,
                "llm_scores": q.llm_scores,
                "llm_score_mean": q.llm_score_mean,
                "llm_score_std": q.llm_score_std,
                "final_score": q.final_score,
                "reasoning": q.reasoning,
                "matched_evidence": q.matched_evidence,
                "missing_evidence": q.missing_evidence,
                "discrepancy": q.discrepancy,
                "uncertain": q.uncertain,
                "needs_review": q.needs_review,
                "calibration_status": q.calibration_status,
            }
            for q in result.questions
        ],
        "created_at": datetime.now(timezone.utc),
    }

    collection.insert_one(doc)

    # Display result
    print("\n====== RESULT ======")
    print(f"Reg No : {student_reg_no}")
    print(f"Status : {status}")
    print(f"Score  : {result.final_score}/100  (embedding: {result.embedding_score})")
    print(f"Grade  : {grade}")

    for q in result.questions:
        label = f"Q{q.qid}" if q.qid != "full" else "Overall"
        spread = f"±{q.llm_score_std}" if q.llm_score_std is not None else "n/a"
        flag = "  ⚠️ review" if q.needs_review else ""
        print(f"\n--- {label}: {q.final_score}/100 "
              f"(embedding {q.embedding_score}, LLM {q.llm_score_mean} {spread}){flag}")
        print(f"  {q.reasoning}")
        if q.matched_evidence:
            print(f"  matched: {'; '.join(q.matched_evidence)}")
        if q.missing_evidence:
            print(f"  missing: {'; '.join(q.missing_evidence)}")

    if result.needs_review:
        print(
            "\n⚠️  One or more questions flagged for human review "
            "(judge/embedding disagreement or unstable self-consistency scores)."
        )


if __name__ == "__main__":
    main()
