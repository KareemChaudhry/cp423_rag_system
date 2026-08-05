import os
import sys
import ollama

from retriever import Retriever, load_chunks

OLLAMA_MODEL = "llama3.2"
TOP_K = 5

SYSTEM_PROMPT = """You are a question-answering assistant. You must answer ONLY using the
provided context chunks below. Do not use any outside knowledge.

Rules:
1. Every claim in your answer must be supported by the context.
2. Cite the chunk(s) you used inline, like [C1] or [C2, C3], right after the claim they support.
3. If the context does not contain enough information to answer the question,
   respond with exactly: "I don't know". Do not guess or use outside knowledge.
4. Keep your answer concise and directly address the question."""

def build_context_block(retrieved_chunks):
    lines = []
    citation_map = {}

    for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
        label = f"C{i}"
        citation_map[label] = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "url": chunk["url"],
        }

        lines.append(f"[{label}] (from \"{chunk['title']}\"): {chunk['text']}")

    return "\n\n".join(lines), citation_map

def build_prompt(query, retrieved_chunks):
    context_block, citation_map = build_context_block(retrieved_chunks)

    user_prompt = f"""Context:
{context_block}

Question: {query}

Answer (cite chunks inline ex. [C1]):"""
    return user_prompt, citation_map

def generate_answer(query, retriever, method="bm25", top_k=TOP_K):

    if method == "bm25":
        retrieved = retriever.bm25_search(query, top_k=top_k)

    elif method == "dense":
        retrieved = retriever.dense_search(query, top_k=top_k)
        
    else:
        raise ValueError(f"Unknown method: {method} (use 'bm25' or 'dense')")

    user_prompt, citation_map = build_prompt(query, retrieved)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer_text = response["message"]["content"]

    return {
        "query": query,
        "method": method,
        "answer": answer_text,
        "citation_map": citation_map,
        "retrieved_chunks": retrieved,
    }


def print_answer(result):
    print(f"\n=== Method: {result['method']} ===")
    print(f"Q: {result['query']}\n")
    print(f"A: {result['answer']}\n")
    print("Citation key:")
    for label, meta in result["citation_map"].items():
        print(f"  [{label}] -> {meta['chunk_id']}  ({meta['title']})")


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    retriever = Retriever(chunks)

    demo_query = "What are the requirements to graduate with the Co-operative Education designation?"

    result_bm25 = generate_answer(demo_query, retriever, method="bm25")
    print_answer(result_bm25)

    result_dense = generate_answer(demo_query, retriever, method="dense")
    print_answer(result_dense)