import json
import os
import ollama

QUESTIONS_FILE = os.path.join("eval", "diagnostic_questions.json")
RESULTS_FILE = os.path.join("eval", "diagnostic_results.json")


OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = """Answer the question directly and concisely using only your own knowledge.
If you do not know the answer, respond with EXACTLY these three words and nothing else:
I don't know"""

def load_questions(path=QUESTIONS_FILE):
    with open(path, encoding="utf-8") as f:

        return json.load(f)

def ask_without_context(question):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        options={"seed": 42, "temperature": 0.0},  # fixed seed + greedy decoding for reproducible results
    )
    return response["message"]["content"]

def run():
    questions = load_questions()
    results = []

    for q in questions:
        print(f"Asking [{q['id']}]: {q['question']}")
        model_answer = ask_without_context(q["question"])

        results.append({
            "id": q["id"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "source_doc": q["source_doc"],
            "model_answer_no_context": model_answer,
        })

        print(f"  Model said: {model_answer[:150]}")
        print(f"  Reference:  {q['reference_answer']}")
        print()

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {RESULTS_FILE}")
    print("\nNext step: manually read through each pair and judge correct/incorrect --")
    print("this judgment call itself is part of what you report in the System Report.")


if __name__ == "__main__":
    run()