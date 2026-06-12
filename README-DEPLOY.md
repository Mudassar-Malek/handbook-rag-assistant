# Deploying the Employee Handbook Assistant (Phase 6)

A FastAPI backend (`api/`) serving the Phase 3 agent over a streaming SSE
endpoint, and a Streamlit chat UI (`ui/`) that talks only to the API.

You can run it **locally (no Docker)** or with **docker compose**.

---

## A. Local (no Docker)

Prereqs: Python 3.12 venv with the project installed, and Phase 1 ingestion
already run (a populated `chroma_db/`).

```bash
# 0. one-time: install deps + ingest the handbook (if not already done)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ingestion.ingest Employee_Handbook_2026.pdf      # builds chroma_db/

# 1. start the API (terminal 1)
uvicorn api.main:app --port 8000
#   -> loads retriever + BM25 + cross-encoder ONCE, then serves.

# 2. start the UI (terminal 2)
export API_BASE_URL=http://localhost:8000
streamlit run ui/app.py
#   -> opens http://localhost:8501
```

Open **http://localhost:8501** and chat.

> No LLM key? It just works: the API auto-falls back to the offline `fake`
> provider (you'll see a startup warning). Set `LLM_PROVIDER=anthropic` +
> `ANTHROPIC_API_KEY` (or `openai` + `OPENAI_API_KEY`) in `.env` for real
> generated answers. Set `LANGFUSE_*` to enable tracing + feedback scoring.

### Local LLM with Ollama (no API key)

Run everything fully offline with [Ollama](https://ollama.com) — no hosted key needed:

```bash
ollama serve                 # start the local model server (:11434)
ollama pull llama3.1         # or qwen2.5 / mistral-nemo (must support structured outputs)

# point the app at Ollama (.env or shell):
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1
# export OLLAMA_BASE_URL=http://localhost:11434   # default
```

This applies everywhere the provider pattern is used: the agent CLI
(`python -m agents.chat --provider ollama`), the API, and the evaluator
(`LLM_PROVIDER=ollama python -m evaluation.run ...`). Smaller models work but
are noticeably slower on CPU — the multi-step pipeline makes 4+ LLM calls per
question, so prefer a GPU/Metal-accelerated host for interactive use. In Docker,
set `OLLAMA_BASE_URL=http://host.docker.internal:11434` so the container can
reach Ollama running on the host.

---

## B. Docker compose

```bash
# 1. ensure the handbook is ingested into ./chroma_db (one-off; see below)
# 2. create the memory file so the bind-mount is a file, not a directory
touch checkpoints.sqlite

# 3. build + start API and UI
docker compose up --build
```

- **UI:** http://localhost:8501  ·  **API:** http://localhost:8000
- `ui` waits for `api` to become **healthy** (`/health` returns 200).
- Put your keys in `.env` (loaded via `env_file`).

### One-off ingestion (not a service)

Ingestion builds `chroma_db/` and should run once, not as a long-lived
container:

```bash
docker compose run --rm api python -m ingestion.ingest data/Employee_Handbook_2026.pdf
```

(Place the PDF where the command can read it; the repo root is the build
context, so `data/Employee_Handbook_2026.pdf` or the repo-root PDF both work.)

---

## Where each volume lives

| Volume | Host path / name | Used by | Notes |
|---|---|---|---|
| Vector store | `./chroma_db` | api (read-only) | Built by ingestion (Phase 1) |
| Thread memory | `./checkpoints.sqlite` | api (read-write) | `touch` it before `up`; the LangGraph `AsyncSqliteSaver` checkpointer |
| Model cache | named volume `model_cache` | api | Cross-encoder + embedding models; survives rebuilds so they aren't re-downloaded |

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/ask` | SSE stream: `status` → `token` → `citations` → `trace` → `done` (rate-limited 10/min/IP) |
| POST | `/feedback` | Push `user_feedback` score (`+1`/`-1`) to Langfuse for a `trace_id` |
| GET | `/health` | Per-component status (ChromaDB, BM25, LLM key, Langfuse); 503 if a critical one is down |
| GET | `/sections` | All `section_number` + `section_title` (UI sidebar index) |
| GET | `/threads/{id}` | Checkpointer message count + history length for a thread |

---

## Tailing agent activity

The API logs structured startup + request info. To watch the agents work:

```bash
# Docker:
docker compose logs -f api

# Local:
# the uvicorn terminal already prints INFO logs (provider, langfuse on/off,
# stream errors). Set tracing on (LANGFUSE_*) to inspect full per-node traces,
# retrieval sub-query spans, and scores in the Langfuse UI.
```

Every `/ask` is one Langfuse trace (session = `thread_id`), with a span per
retrieval sub-query carrying the full per-chunk score breakdown — the fastest
way to see *why* a given answer was produced.
