import os
import re
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = os.path.join("data", "chunks", "chunks.jsonl")
EMBEDDINGS_CACHE = os.path.join("data", "chunks", "dense_embeddings.npy")

DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def load_chunks(path=CHUNKS_FILE):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]

        tokenized_corpus = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        self.chunk_embeddings = self._load_or_build_embeddings()

    def _load_or_build_embeddings(self):
        if os.path.exists(EMBEDDINGS_CACHE):
            return np.load(EMBEDDINGS_CACHE)

        embeddings = self.dense_model.encode(
            self.texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        os.makedirs(os.path.dirname(EMBEDDINGS_CACHE), exist_ok=True)
        np.save(EMBEDDINGS_CACHE, embeddings)
        return embeddings

    def bm25_search(self, query, top_k=TOP_K):
        scores = self.bm25.get_scores(tokenize(query))
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]

    def dense_search(self, query, top_k=TOP_K):
        query_vec = self.dense_model.encode([query], normalize_embeddings=True)[0]
        scores = self.chunk_embeddings @ query_vec
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]


def print_results(label, results):
    print(f"\n--- {label} ---")
    for rank, (chunk, score) in enumerate(results, 1):
        print(f"#{rank}  score={score:.3f}  [{chunk['chunk_id']}]  ({chunk['title']})")
        print("    ", chunk["text"][:160].replace("\n", " "), "...")


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    retriever = Retriever(chunks)

    demo_query = "co-op work term requirements"
    print(f'\nQuery: "{demo_query}"')

    print_results("BM25 results", retriever.bm25_search(demo_query))
    print_results("Dense results", retriever.dense_search(demo_query))