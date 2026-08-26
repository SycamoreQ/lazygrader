# Automated Answer Evaluation System (AI-Based)

An AI-powered system to automatically evaluate handwritten student answers against model answers using OCR, semantic similarity, and NLP techniques.

## 🚀 Features
- Extracts handwritten text from PDF answer sheets using **Mistral OCR**
- Computes **semantic similarity** between student and model answers
- Generates **scores out of 100**, grades, and **human-like feedback**
- Automatically stores evaluation results in **MongoDB Atlas**
- Supports multi-page PDFs and multiple students

## 📂 Contents

- `model ans.pdf` – Model answer PDF  
- `stu hw ans.pdf` – Example student answer PDF  
- `evapdb.py` – Main evaluation script  
- `db-test.py` – MongoDB connection test  
- Other scripts (OCR experiments)  
- Supporting PDFs and test files  
- GitHub repo: https://github.com/sriks2112/automated-answer-evaluation-system

## 🧠 Algorithms & Models Used
- **OCR**: Mistral OCR (`mistral-ocr-latest`)
- **Text Embeddings**: Sentence-BERT (`all-MiniLM-L6-v2`)
- **Similarity Metric**: Cosine Similarity
- **Evaluation Logic**: Rule-based grading with AI-assisted feedback

## 🛠 Tech Stack
- **Python**
- **Mistral AI (OCR API)**
- **Sentence Transformers (SBERT)**
- **MongoDB Atlas**
- **Git & GitHub**

## 📄 Workflow
1. Upload Model Answer PDF
2. Upload Student Answer PDF
3. Extract text using AI OCR
4. Compare answers using semantic similarity
5. Generate score, grade, and feedback
6. Store results in database automatically

## 🚀 How to Run

### 1. Clone repository

git clone https://github.com/sriks2112/automated-answer-evaluation-system.git
cd automated-answer-evaluation-system

### 2. Install dependencies
pip install -r requirements.txt
Or manually install:
pip install mistralai sentence-transformers pymongo pytz

###3. Set up environment

Create .env file (recommended) with:
MISTRAL_API_KEY=your_api_key_here
MONGO_USER=your_mongo_user
MONGO_PASS=your_mongo_password

###4. Run evaluation

python evapdb.py
input the student register number when prompted.

## 📊 Result Format

Example output:

====== RESULT ======

Reg No : CS202401

Status : PASS

Score  : 81/100

Grade  : A

Feedback:Very good answer. Minor improvements can make it perfect.

Results are stored in MongoDB Atlas.

## 🧠 How It Works

OCR Extraction:
Uses Mistral AI OCR to extract text from PDF images.

Semantic Similarity:
Uses Sentence-BERT embeddings and cosine similarity for text comparison.

Scoring:
Similarity × 100 → Evaluated score → mapped to grade.

Feedback:
Generates human-like feedback based on similarity and template logic.

Database Storage:
Stores every evaluation in MongoDB Atlas with timestamp.

## 📁 Database Structure

Each stored document contains:

Field	          ->        Description

student_reg_no	->   Student registration number

score	          ->   Final score out of 100

grade           ->	 Grade (O/A/B etc.)

status	        ->   PASS / FAIL

feedback	      ->   Human-like feedback

similarity	    ->   Similarity score (0–1)

created_at	    ->   Evaluation timestamp

## 🧭 LLM Calibration Layer (`calibrator.py`)

The SBERT cosine-similarity score is a fast first pass, but it can't say
*why* it gave a score, and it can be fooled by paraphrasing or by an answer
that's mostly right but missing one key idea. `calibrator.py` adds a second,
independent grader:

1. An LLM (`mistral-large-latest` by default) reads the model answer and the
   student answer and produces its own 0–100 score, a short natural-language
   explanation, and lists of strengths/gaps.
2. The final score is a 50/50 blend of the embedding score and the LLM
   score — a lightweight ensemble rather than trusting either grader alone.
3. If the two scores disagree by more than 20 points, the result is flagged
   `needs_review` so a human grader can take a look, instead of the
   disagreement being silently averaged away.
4. If the LLM call fails or returns unparseable output, the pipeline falls
   back to the embedding score automatically — grading never breaks because
   the calibrator is unavailable.

The explanation and strengths/gaps are now stored per-submission in MongoDB
(`llm_score`, `embedding_score`, `strengths`, `gaps`, `discrepancy`,
`needs_review`, `calibration_status`) and printed alongside the grade.

## 🧩 Future Improvements

✔ Per-question matching
✔ Partial credit rubric
✔ Web interface (FastAPI / React)
✔ Student performance analytics dashboard
✔ Persist `needs_review` submissions to a separate queue for instructors

##  📞 Contact

If you’d like to know more or collaborate:

GitHub: https://github.com/sriks2112

Email: srinithiks002@gmail.com
