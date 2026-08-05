# CP423 Course Project: RAG System over the WLU Academic Calendar

A Retrieval-Augmented Generation system built over the Wilfrid Laurier University Undergraduate Academic Calendar (2025/2026 and 2026/2027). 

Combines BM25 (classical) and sentence-transformer (dense) retrieval with a locally-run LLM (Llama 3.2 via Ollama) for cited question answering.

## Team
```
Tiara Bhakat - 169023019
Kareem Chaudhry - 169035228
``` 

## Project Structure

```
cp423_rag_system/
├── data/
│   ├── raw/          # Scraped Laurier HTML/text docs
│   └── chunks/        # Chunked text JSON files
├── eval/
│   ├── diagnostic_questions.json  # 10 no-context diagnostic questions
│   ├── diagnostic_results.json    # diagnostic experiment results
│   ├── eval_set.json      # 11 gold-standard question/answers
│   └── eval_results.json  # both methods
├── src/
│   ├── scraper.py     # Corpus collection script
│   ├── chunker.py     # Text processing & chunking
│   ├── retriever.py   # BM25 + Dense vector search
│   ├── pipeline.py    # LLM prompt & citation generator
│   ├── diagnostic.py  # runs the corpus-suitability diagnostic
│   └── evaluate.py    # runs the evaluation
├── .gitignore
├── README.md
├── requirements.txt
└── run_it_all.py       # Single command execution script
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** -- install from [ollama.com/download](https://ollama.com/download), then pull the model used in this project:

Ollama must be running in the background (it starts automatically after install).

## Setup

`python -m pip install -r requirements.txt`

The first run that uses dense retrieval will download the `all-MiniLM-L6-v2`sentence-transformers model (~90MB) from Hugging Face automatically.

## Reproducing All Results

`python run_it_all`

This runs, in order:
1. `src/chunker.py` -- chunks the corpus (`data/raw/` -> `data/chunks/chunks.jsonl`)
2. `src/diagnostic.py` -- runs the 10 no-context diagnostic questions (`eval/diagnostic_results.json`)
3. `src/evaluate.py` -- runs the full 11-question gold-standard evaluation across both
   retrieval methods (`eval/eval_results.json`)

All generation calls use a **fixed seed (42) and temperature 0** for reproducible output.

Total runtime is roughly 15-20 minutes on a laptop CPU most of it spent on LLM generation calls.

## Re-collecting the Corpus (optional)

The scraped corpus is already included in `data/raw/`, so this step is **not required** to reproduce the report's results. To re-scrape from the live site:

`python src/scraper.py`

Note: this depends on the live `academic-calendar.wlu.ca` site and its current content, so it is not guaranteed to reproduce byte-identical output to what's already in `data/raw/`.

## Corpus

- **Source:** Wilfrid Laurier University Undergraduate Academic Calendar (academic-calendar.wlu.ca)
- **Size:** 500 documents, chunked into ~3,175 retrieval chunks (80 words/chunk, 20-word overlap)
- **Why this corpus:** it is specific, year-versioned institutional content (dean names,course codes, program requirements, deadlines) that general-purpose LLMs are unlikely to have memorized. This is confirmed by the diagnostic experiment (see report):

    the base model answered 0/10 diagnostic questions correctly without retrieval, and in one run hallucinated a specific wrong answer (see `eval/diagnostic_results.json`).

## System Design

- **Retrieval:** BM25 (`rank-bm25`) and dense embeddings (`sentence-transformers`, model `all-MiniLM-L6-v2`), both indexed over the same chunked corpus.
- **Generation:** `llama3.2` (3B) via a local Ollama server, prompted to answer only from retrieved context, cite chunks inline (e.g. `[C1]`), and say "I don't know" when context is insufficient.
- **Evaluation:** 11 hand-written, human-verified questions (5 factoid, 3 multi-hop, 3 unanswerable), each with a reference answer and ground-truth source document(s). Retrieval is scored automatically (doc-level recall); generation quality is judged manually per the assignment's requirements (correctness, citation support appropriate "I don't know" behavior).

See the System Report (PDF) for full results, error analysis, and limitations.
