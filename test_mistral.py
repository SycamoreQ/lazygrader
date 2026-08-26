import os
import json

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv(override=True)

key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=key)

prompt = """
You are an impartial exam grader.

You will be shown a MODEL ANSWER (reference/correct answer) and a STUDENT ANSWER
(extracted via OCR, so it may contain minor transcription errors — do not penalize for those)
for a single question.

MODEL ANSWER:
A type of machine learning where the model is trained on labeled data with known inputs and outputs.

STUDENT ANSWER:
Supervised learning is a machine learning algorithm which we give labeled data as input
and expect output as prediction we want.

Judge the student answer on its actual content and reasoning, not just surface wording
overlap. Ground your judgment in the text: quote short exact phrases from the two answers
to support your score.

Return ONLY a JSON object with this exact shape, nothing else:

{
  "score": <integer 0-100>,
  "reasoning": "<2-3 sentence explanation of the score, written for the student>",
  "matched_evidence": ["<short exact phrase from the student answer that matches the model answer>"],
  "missing_evidence": ["<short exact phrase or idea from the model answer the student answer lacks>"]
}
"""

response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    response_format={"type": "json_object"},
    temperature=0.7,
)

raw = response.choices[0].message.content

print("\nRAW RESPONSE:")
print(raw)

print("\nPARSED:")
try:
    parsed = json.loads(raw)
    print(json.dumps(parsed, indent=2))

    print("\nSCORE:")
    print(parsed["score"])

except Exception as exc:
    print("\nPARSE ERROR:")
    print(type(exc).__name__, exc)