from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv
import os, json

# -------- Load env + setup client --------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

client = genai.Client(api_key=api_key)

app = Flask(__name__)
CORS(app)  # allow requests from extension

# -------- Helper: build prompt --------
def build_prompt(title: str, content: str) -> str:
    return f"""
You are a study assistant.

Given the following webpage titled "{title}", generate ALL of the following in JSON:

1. A short, crisp summary in 4-6 sentences ONLY (max 100 words).
2. Possible interview-style open-ended questions based fully on the webpage content.
3. 20 multiple-choice questions (MCQs), each with:
   - question
   - 4 options (A,B,C,D)
   - correct_option
   - explanation for the correct option
4. 15 flashcards with:
   - front (term / question)
   - back (explanation)

Return ONLY valid JSON in this exact structure:

{{
  "summary": "....",
  "interview_questions": ["...", "..."],
  "mcqs": [
    {{
      "question": "...",
      "options": [...],
      "correct_option": "A",
      "explanation": "Reason why the correct answer is correct."
      }}
  ],
  "flashcards": [
    {{ "front": "...", "back": "..." }}
  ]
}}

CONTENT:
--------------------
{content}
--------------------
"""

# -------- API route --------
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True) or {}
        title = data.get("title", "")
        content = data.get("content", "")

        print("\n--- RECEIVED REQUEST ---")
        print("TITLE:", title)
        print("CONTENT:", content[:100], "...")

        if not content.strip():
            print("ERROR: No content received")
            return jsonify({"error": "No content provided"}), 400

        prompt = build_prompt(title, content)

        print("\n--- SENDING TO GEMINI ---")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        print("\n--- RAW GEMINI RESPONSE ---")
        print(response.text)

        text = response.text or ""

        clean = text.replace("```json", "").replace("```", "").strip()

        print("\n--- CLEANED JSON STRING ---")
        print(clean)

        parsed = json.loads(clean)

        print("\n--- SUCCESS: RETURNING PARSED JSON ---")
        return jsonify(parsed)

    except json.JSONDecodeError:
        print("\n--- ERROR: JSON PARSE FAILED ---")
        print("RAW RESPONSE:", text)
        return jsonify({"error": "Invalid JSON from model", "raw": text}), 500

    except Exception as e:
        print("\n--- UNEXPECTED ERROR ---")
        print(str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "Study backend is running"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
