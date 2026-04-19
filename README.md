# 🧠 Semantic Drift RAG

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![LangGraph](https://img.shields.io/badge/LangGraph-1.1.6-purple) ![License](https://img.shields.io/badge/License-MIT-yellow)

A strict, grounded, and self-correcting RAG system built with FastAPI, LangGraph, FAISS, and BM25. Goes beyond standard retrieval-augmented generation by addressing both major sources of LLM hallucination — not just retrieval quality, but whether the query was answerable from the uploaded documents in the first place.

---

## 📋 Table of Contents

- [The Problem](#the-problem)
- [Features](#features)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Key Design Decisions](#key-design-decisions)
- [License](#license)

---

## The Problem

Most RAG systems have two distinct failure modes:

**1. Hallucination within the documents** — the retrieval isn't accurate enough, or the model struggles with complex queries and fills in the gaps with invented information.

**2. Hallucination outside the documents** — the user asks something the uploaded documents don't cover, and the model confidently answers anyway using its own general knowledge instead of saying it doesn't know.

The second failure mode is largely unaddressed in standard RAG pipelines. This system solves it with a **pre-generation scope detection layer**, and addresses the first with a **self-correcting LangGraph evaluation loop**.

---

## Features

### Core Features

| Feature                | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| Multi-format ingestion | Supports `.pdf`, `.txt`, URLs, directories, and `.zip` archives |
| Hybrid retrieval       | FAISS (dense, 70%) + BM25 (sparse, 30%) ensemble retriever      |
| Self-correcting loop   | LangGraph pipeline with evaluator that retries up to 3 times    |
| Source citations       | Every answer includes `(Source: 'filename', page N)` references |
| Per-user isolation     | Each user has their own FAISS index and session state           |
| Rate limiting          | 15 requests/minute protection via SlowAPI                       |

### Drift Detection

| Feature             | Description                                                                |
| ------------------- | -------------------------------------------------------------------------- |
| Adaptive thresholds | Derived from Z-score of actual corpus distribution — never hardcoded       |
| Auto-updating       | Thresholds recompute automatically when new documents are uploaded         |
| Session trajectory  | Tracks conversation direction over last 20 queries, not just current query |
| Three-way decision  | Refuse / ask clarification / answer based on drift score                   |
| File-safe sessions  | Session state persisted with file locking for concurrent request safety    |

---

## System Architecture

The system consists of:

- **API Server (FastAPI)** — Handles document uploads, query answering, and session management
- **VectorStore** — Builds and loads FAISS + BM25 hybrid retriever, computes corpus centroid and Z-score stats on upload
- **DriftDetector** — Pre-generation scope check using corpus centroid, session centroid, and adaptive Z-score thresholds
- **LangGraph Pipeline** — Retriever → Responder → Evaluator loop with conditional retry routing
- **Disk Storage** — Per-user FAISS index, `corpus_centroid.npy`, `corpus_stats.npy`, and `session_state.json`

**[View System Architecture Diagram](diagrams/system-architecture.png)**

---

## How It Works

### Document Upload

Documents are processed in five steps when uploaded:

1. Loaded and chunked with `RecursiveCharacterTextSplitter` (500 tokens, 50 overlap)
2. Embedded with `all-MiniLM-L6-v2` and indexed into FAISS
3. BM25 index built from the same chunks and serialized to `documents.json`
4. Corpus centroid computed as the normalized mean of all chunk embeddings
5. Mean (μ) and standard deviation (σ) of cosine distances computed and saved as `corpus_stats.npy`

The adaptive thresholds are set as:

- Warning threshold: `μ + 2.5σ`
- Drift threshold: `μ + 3.5σ`

These thresholds are not hardcoded — they reflect the actual semantic density of whatever was uploaded. Upload five documents, get thresholds for those five. Upload two more and the thresholds update automatically.

### Drift Detection (Pre-Generation Check)

Before the LLM is called for any query:

1. Query is embedded and its cosine distance from the corpus centroid is computed (`query_drift_score`)
2. If there are 2+ previous queries in the session, a session centroid is computed. Its distance from the corpus centroid gives the `trajectory_drift_score` — where the conversation is heading, not just where the current query sits
3. `final_score = max(query_drift_score, trajectory_drift_score × 0.7)`
4. Final score is compared against the adaptive thresholds:
   - `final_score >= drift_threshold` → **refuse** — LLM is not called, out-of-scope message returned
   - `final_score >= warning_threshold` → **ask_clarification** — user is prompted to rephrase
   - below both → **answer** — proceed to the LangGraph pipeline

### LangGraph Self-Correcting Loop

For queries that pass the drift check:

```
retriever → responder → evaluator → (router)
                ↑                       |
                └── retry if score < 0.8 (max 3 retries)
```

- **Retriever** — fetches top-4 chunks using the hybrid FAISS + BM25 ensemble. On retry, uses the evaluator's refined query
- **Responder** — Qwen3 80B generates a grounded answer with source citations. Previous evaluator suggestions are injected into the prompt on retry
- **Evaluator** — Llama 3.3 70B scores the answer (0–1) for faithfulness and relevance using structured output. Produces a `score`, `suggestion`, and `refined_query`
- **Router** — exits the loop when `score >= 0.8` or after 3 retries, whichever comes first

### Task Lifecycle

```
POST /upload → [chunk → embed → FAISS + BM25 → centroid + stats] → ready

POST /generate-answer
    → DriftDetector.analyze()
        → refuse        (score >= high_threshold)
        → clarify       (score >= mid_threshold)
        → LangGraph     (score below both)
            → retrieve → respond → evaluate → route
                                        ↑______| retry
```

---

## Tech Stack

| Component              | Technology                                  |
| ---------------------- | ------------------------------------------- |
| API framework          | FastAPI + SlowAPI                           |
| Embeddings             | `all-MiniLM-L6-v2` via HuggingFace          |
| Dense retrieval        | FAISS                                       |
| Sparse retrieval       | BM25                                        |
| Pipeline orchestration | LangGraph                                   |
| Generation model       | Qwen3 80B via OpenRouter                    |
| Evaluation model       | Llama 3.3 70B via Groq                      |
| Drift detection        | Custom — Z-score normalized cosine distance |
| File locking           | `filelock`                                  |
| Data validation        | Pydantic v2                                 |
| Python version         | 3.10+                                       |

---

## Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **Git** — [Download](https://git-scm.com/)
- **Groq API key** — [console.groq.com](https://console.groq.com)
- **OpenRouter API key** — [openrouter.ai](https://openrouter.ai)
- **Hugging Face token** — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hrishikeshChandi/semantic-drift-rag.git
cd semantic-drift-rag
```

### 2. Create Virtual Environment and install the dependencies

```bash
uv sync
source .venv/bin/activate # Linux/macOS

# .venv\Scripts\activate # Windows
```

### 3. Configure environment

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
HF_TOKEN=your_huggingface_token

MODULE = "main:app"
HOST = "127.0.0.1"
PORT = 8000

LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/app.log
```

---

## Configuration

| Variable             | Default     | Description                             |
| -------------------- | ----------- | --------------------------------------- |
| `HOST`               | `127.0.0.1` | Server host                             |
| `PORT`               | `8000`      | Server port                             |
| `MODULE`             | `main:app`  | Uvicorn module string                   |
| `UPLOAD_ROOT`        | `data`      | Root directory for user data            |
| `GROQ_API_KEY`       | —           | Groq API key for evaluator model        |
| `OPENROUTER_API_KEY` | —           | OpenRouter API key for generation model |
| `HF_TOKEN`           | —           | Hugging Face token for embeddings model |

---

## Running the System

1. Start the backend:

   ```bash
   uv run main.py
   ```

2. Start the frontend:

   ```bash
   cd frontend/
   python -m http.server 8080
   ```

API docs available at `http://localhost:8000/docs`

---

## API Endpoints

### Documents

| Method   | Endpoint           | Description                           | Rate Limit |
| -------- | ------------------ | ------------------------------------- | ---------- |
| `POST`   | `/files/upload`    | Upload documents and build index      | 15/min     |
| `GET`    | `/files/{user_id}` | List uploaded files for a user        | Unlimited  |
| `DELETE` | `/files/{user_id}` | Delete all files and index for a user | 15/min     |

### Queries

| Method   | Endpoint             | Description                                     | Rate Limit |
| -------- | -------------------- | ----------------------------------------------- | ---------- |
| `POST`   | `/generate-answer`   | Query documents with drift detection            | 15/min     |
| `DELETE` | `/session/{user_id}` | Reset session memory without deleting documents | Unlimited  |

---

## Testing

### Upload a document

```bash
curl -X POST http://localhost:8000/files/upload \
  -F "user_id=user123" \
  -F "files=@document.pdf"
```

### Ask an in-scope question

```bash
curl -X POST http://localhost:8000/generate-answer \
  -F "user_id=user123" \
  -F "query=What is the main argument of chapter 3?"
```

### Ask an out-of-scope question (should be refused)

```bash
curl -X POST http://localhost:8000/generate-answer \
  -F "user_id=user123" \
  -F "query=What is the capital of France?"
```

### List uploaded files

```bash
curl http://localhost:8000/files/user123
```

### Reset session memory

```bash
curl -X DELETE http://localhost:8000/session/user123
```

---

## Project Structure

```
semantic-drift-rag/
├── config/
│   └── constants.py               # Env vars, shared embeddings model singleton
├── core/
│   └── limiter.py                 # Rate limiter setup
│   └── logging_config.py          # Logger setup
├── document_ingestion/
│   └── processor.py               # DocumentProcessor — PDF, TXT, URL, directory loading and chunking
├── drift_detector/
│   └── detector.py                # DriftDetector — centroid tracking, Z-score thresholds, session memory
├── frontend/                      # Frontend files
│   └── index.html
│   └── script.js
│   └── style.css
├── graph_builder/
│   └── builder.py                 # LangGraph pipeline construction
├── llm/
│   └── llm.py                     # LLM initialization — Qwen 80B + Llama 70B
├── models/
│   └── model.py                   # Pydantic schema for evaluator structured output
├── nodes/
│   └── nodes.py                   # Retriever, Responder, Evaluator node logic
├── routers/
│   └── files.py                   # /upload, /files endpoints
├── state/
│   └── rag_state.py               # LangGraph state schema (Pydantic)
├── vectorstore/
│   └── vectorstore.py             # VectorStore — FAISS + BM25 hybrid retriever, centroid/stats computation
├── main.py                        # FastAPI entry point, /generate-answer and /session endpoints
```

---

## Key Design Decisions

- **Why Z-score thresholds instead of hardcoded values?**

  The semantic space of embedding models like all-MiniLM-L6-v2 is not uniformly distributed. A fixed threshold (e.g., 0.35) may fail across different datasets — either never triggering or over-triggering depending on corpus density.

  Instead, this system computes the distribution of embedding distances from the corpus centroid and derives thresholds using Z-score based calibration:
  - μ + 2.5σ → boundary (warning zone)
  - μ + 3.5σ → out-of-scope (hard rejection)

  This creates a soft decision boundary:
  - Queries near the boundary are still allowed (with warnings)
  - Only clearly out-of-distribution queries are blocked

  As new documents are uploaded, μ and σ are recomputed, allowing the system to self-calibrate dynamically to the dataset.

- **Why track session trajectory?**

  A single query may be borderline — close enough to the corpus that it passes individually. However, a sequence of queries may gradually drift away from the document domain.

  To capture this, the system maintains a session-level embedding history and computes a session centroid, representing the overall direction of the conversation.

  Drift is then computed using two signals:
  - Query drift → distance of current query from corpus centroid
  - Trajectory drift → distance of session centroid from corpus centroid

  These are combined as:

  `final_score = max(query_drift, trajectory_drift × 0.7)`

  The trajectory signal is down-weighted (0.7) to ensure it influences the decision without overpowering the current query, since historical drift is a weaker signal than immediate query relevance.

  This allows the system to detect gradual topic drift, even when individual queries appear valid.

- **Why two different LLM providers?**

  Generation runs once per query and can tolerate latency — Qwen3 80B on OpenRouter gives strong instruction-following and citation quality. Evaluation can run up to 3 times per query in the retry loop, so it needs to be fast — Llama 3.3 70B on Groq's LPU hardware keeps each evaluation call under 1 second so it doesn't dominate total response time.

- **Why a hybrid retriever?**

  Dense retrieval (FAISS) is strong for semantic similarity but misses exact keyword matches — technical terms, acronyms, proper nouns. BM25 handles exact matches well but misses semantic similarity. The 70/30 ensemble captures both signals without needing to choose.

---

## License

This project is licensed under the MIT License.
