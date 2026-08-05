import os
import json

from retriever import Retriever, load_chunks
from pipeline import generate_answer

EVAL_SET_FILE = os.path.join("eval", "eval_set.json")
RESULTS_FILE = os.path.join("eval", "eval_results.json")

TOP_K = 5

def load_eval_set(path=EVAL_SET_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def retrieved_doc_ids(retrieved_chunks):
    return {chunk["doc_id"] for chunk, score in retrieved_chunks}


def source_doc_ids_for(question):
    ids = set()
    for s in question.get("source_doc_ids", []):
        ids.add(s.split("_chunk_")[0])
    return ids

def score_retrieval(question, retrieved_chunks):
    expected = source_doc_ids_for(question)
    if not expected:
        return None
    actual = retrieved_doc_ids(retrieved_chunks)
    hit_docs = expected & actual
    return {
        "expected_docs": sorted(expected),
        "retrieved_docs": sorted(actual),
        "recall": len(hit_docs) / len(expected),
    }


def run():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")
    retriever = Retriever(chunks)

    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} eval questions.")

    all_results = []

    for q in eval_set:
        print(f"\n=== [{q['id']}] ({q['type']}) {q['question']}")

        question_result = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "methods": {},
        }

        for method in ["bm25", "dense"]:
            gen_result = generate_answer(q["question"], retriever, method=method, top_k=TOP_K)
            retrieval_score = score_retrieval(q, gen_result["retrieved_chunks"])

            question_result["methods"][method] = {
                "generated_answer": gen_result["answer"],
                "retrieval_score": retrieval_score,
                "retrieved_chunk_ids": [c["chunk_id"] for c, s in gen_result["retrieved_chunks"]],
                "manual_judgment": {
                    "correct": None,
                    "supported_by_citation": None,
                    "notes": ""
                }
            }

            print(f"  [{method}] retrieval recall: {retrieval_score['recall'] if retrieval_score else 'N/A (unanswerable)'}")
            print(f"  [{method}] answer: {gen_result['answer'][:150]}")

        all_results.append(question_result)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results for {len(all_results)} questions to {RESULTS_FILE}")


if __name__ == "__main__":
    run()