# Architecture & Technology Stack

The **Employee Handbook Assistant** is a production-grade Retrieval-Augmented
Generation (RAG) system that answers HR questions about a single document —
`Employee_Handbook_2026.pdf` — with exact, verifiable section citations like
`[5.1 Earned Leave (EL), p.24]`.

It is built as **six layered phases**, each consuming the one below it. This
document explains the overall architecture, how data flows through the system,
and every technology used and *why*.

---

## 1. System at a glance

```
                          ┌─────────────────────────────────────────────┐
                          │            Langfuse observability            │  ← wraps everything (Phase 4)
                          └─────────────────────────────────────────────┘
   ┌────────────┐   ┌────────────┐   ┌──────────────────┐   ┌──────────────────────┐
   │ Phase 1    │   │ Phase 2    │   │ Phase 3          │   │ Phase 6              │
   │ Ingestion  │──▶│ Retrieval  │──▶│ Agent (LangGraph)│──▶│ FastAPI  +  Streamlit│
   │ PDF→chunks │   │ hybrid     │   │ router→…→grader  │   │ /ask (SSE)   chat UI │
   └────────────┘   └────────────┘   └──────────────────┘   └──────────────────────┘
         │                │                   │
         ▼                │                   ├──────────▶ Phase 3.5  MCP tool (Cursor)
   ┌────────────┐         │                   └──────────▶ Phase 5    Evaluation (RAGAS)
   │  ChromaDB  │◀────────┘  (single source of truth: vectors + chunk metadata)
   └────────────┘
```

| Phase | Layer | Package(s) | Responsibility |
|------:|-------|------------|----------------|
| 1 | `ingestion/` | PyMuPDF, ChromaDB, Sentence-Transformers | PDF → clean, heading-aware chunks → vector store |
| 2 | `retrieval/` | rank-bm25, Sentence-Transformers (cross-encoder) | question → ranked sections (hybrid search) |
| 3 | `agents/` | LangGraph, LangChain, Pydantic | conversational multi-agent assistant |
| 3.5 | `mcp_server.py` | MCP (FastMCP) | expose retriever as a Cursor tool |
| 4 | `observability.py` | Langfuse | tracing, scoring, prompt management |
| 5 | `evaluation/` | RAGAS, LangChain-HuggingFace | golden dataset + metrics + chunking A/B |
| 6 | `api/`, `ui/` | FastAPI, Streamlit, SSE, Docker | streaming API + chat UI + deployment |

---

## 2. Technology stack

### Core RAG infrastructure

| Technology | Version | Where | Why this choice |
|------------|---------|-------|-----------------|
| **Python** | 3.11–3.13 | everywhere | ChromaDB doesn't support 3.14 yet; 3.12 is the tested target. |
| **PyMuPDF** (`fitz`) | ≥1.24 | `ingestion/extract.py` | Fast, accurate PDF text + table extraction. Gives per-page control and a `find_tables()` fallback for the two tables that matter (revision history, leave entitlement). |
| **ChromaDB** | ≥0.5 | `ingestion/store.py`, `retrieval/loader.py` | Local, embedded, persistent vector DB (no server to run). Cosine space. Acts as the **single source of truth** — both vectors *and* chunk metadata live here, so Phase 2 rebuilds its BM25 index from the same data. |
| **Sentence-Transformers** | 3.0–3.3 | embeddings + reranker | Local embedding model (`all-MiniLM-L6-v2`, no API key) and the cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`). |
| **rank-bm25** | ≥0.2.2 | `retrieval/bm25.py` | Classic lexical search to catch verbatim acronyms (`EL`, `LOP`, `POSH`) that dense embeddings miss. |
| **NumPy / Transformers / Torch** | numpy<2, transformers 4.40–4.45 | reranker stack | Pinned to a known-good set: torch 2.2.x needs numpy<2, transformers 5.x is incompatible. |

### Agent orchestration (Phase 3)

| Technology | Version | Where | Why this choice |
|------------|---------|-------|-----------------|
| **LangGraph** | 0.2.40–0.2.x | `agents/graph.py` | A `StateGraph` models the assistant as explicit nodes + conditional edges (router → rewriter → retriever → answer → grader). Gives us a real **retry loop** (grader → rewriter) and a typed state object — far more controllable than a linear chain. |
| **langgraph-checkpoint-sqlite** | 2.x | `agents/graph.py`, `api/main.py` | Per-thread **memory**. `SqliteSaver` (sync, CLI) and `AsyncSqliteSaver` (async, API) persist conversation state keyed by `thread_id`, so follow-ups resolve against history. |
| **LangChain Core** | 0.3.x | `agents/llm.py`, `state.py` | Message types (`HumanMessage`/`AIMessage`/`SystemMessage`) and the `.with_structured_output()` interface used for every agent decision. |
| **Pydantic** | v2 (via langchain-core) | `agents/schemas.py` | Structured outputs: `RouteDecision`, `RewriteOutput`, `GradeOutput`. The LLM is *forced* to return validated, typed fields — no fragile string parsing. |

### LLM providers (pluggable)

All providers implement one internal interface (`agents/llm.py`), selected by
`LLM_PROVIDER` in `.env`. This is the key to running the **same graph** with a
hosted model, a local model, or no model at all.

| Provider | Package | Needs key? | Notes |
|----------|---------|:---------:|-------|
| **Anthropic** | `langchain-anthropic` 0.3.x | yes | Claude (default `claude-3-5-sonnet-latest`). |
| **OpenAI** | `langchain-openai` 0.3.x | yes | GPT (default `gpt-4o-mini`). |
| **Ollama** | `langchain-ollama` | **no** | Local models (`llama3.1`, `qwen2.5`, …) via `ollama serve`. Uses JSON-schema structured outputs so the router/grader still return valid Pydantic on local models. Fully offline, free. |
| **fake** | `agents/fake_llm.py` | **no** | Deterministic, rule-based stand-in implementing the same interface. Runs the entire graph offline in CI; composes answers from the *real* retrieved excerpts. |

### Observability (Phase 4)

| Technology | Version | Where | Why this choice |
|------------|---------|-------|-----------------|
| **Langfuse** | v3 (3.x) | `observability.py` | LLM-native tracing: nested spans per node/LLM/tool call, custom **scores** (grader grounded/complete, retrieval confidence, user feedback), **prompt management** (the answer prompt is editable in the UI without a redeploy), and cost/latency tracking. **Degrades gracefully to a no-op** when `LANGFUSE_*` env vars are unset. |
| **langchain** (umbrella) | 0.3.x | callback handler | Required by Langfuse's LangChain callback handler that auto-instruments the graph. |

### Evaluation (Phase 5)

| Technology | Version | Where | Why this choice |
|------------|---------|-------|-----------------|
| **RAGAS** | 0.2.x | `evaluation/ragas_eval.py` | Standard RAG metrics: `faithfulness`, `answer_relevancy`, `answer_correctness` (full pipeline) and `context_precision`, `context_recall` (retrieval). Also `TestsetGenerator` for synthetic questions. |
| **langchain-huggingface** | 0.1.x | `evaluation/judge.py` | Local HF embeddings for RAGAS so evaluation never makes **paid embedding calls**. |
| **tiktoken** | ≥0.7 | `evaluation/recursive_baseline.py` | Token-accurate `RecursiveCharacterTextSplitter` for the Strategy-B chunking baseline. |
| (custom) | — | `evaluation/metrics.py` | `section_hit_rate` — an **LLM-free** metric (`ground_truth_sections ⊆ retrieved`), the most interpretable number in the project; runs with zero keys. |

### API, UI & deployment (Phase 6)

| Technology | Version | Where | Why this choice |
|------------|---------|-------|-----------------|
| **FastAPI** | ≥0.110 | `api/main.py` | Async backend. **Lifespan** loads the heavy models (retriever, BM25, cross-encoder, compiled graph) exactly once, never per request. |
| **Uvicorn** | ≥0.29 | runtime | ASGI server. |
| **sse-starlette** | ≥2.0 | `api/main.py` | Server-Sent Events to stream `/ask`: `status → token → citations → trace → done`. |
| **slowapi** | ≥0.1.9 | `api/main.py` | Per-IP rate limiting (10/min on `/ask`). |
| **aiosqlite** | 0.20.x | async checkpointer | Pinned `<0.21` because 0.21 removed `Connection.is_alive`, which `langgraph-checkpoint-sqlite` 2.x calls. |
| **httpx** | ≥0.27 | `ui/client.py`, tests | HTTP/SSE client for the UI and the API tests. |
| **Streamlit** | ≥1.36 | `ui/app.py` | Single-page chat UI that talks *only* to the API. Surfaces every Phase 1–5 feature: citations, agent trace, memory, feedback. |
| **Docker / docker-compose** | — | `api/Dockerfile`, `ui/Dockerfile` | Multi-stage builds, non-root users, volume management for `chroma_db`, checkpoints, and model cache. |

### Integration

| Technology | Version | Where | Why this choice |
|------------|---------|-------|-----------------|
| **MCP** (Model Context Protocol, FastMCP) | ≥1.2 | `mcp_server.py` | Exposes the **retriever** (not the full agent) as a Cursor tool `search_handbook`. Cursor's own model is the LLM, so no API key is needed — the tool just returns ranked excerpts + grounding rules. |
| **pydantic-settings** | ≥2.0 | every `config.py` | Typed configuration loaded from `.env` with sane defaults for every knob. |

---

## 3. Phase-by-phase architecture

### Phase 1 — Ingestion (`ingestion/`)

**Goal:** turn a messy 2–3 column PDF into clean, citable chunks. Core design
bet (later proven in Phase 5): **heading-aware chunking beats naive fixed-size
splitting** for a structured policy document.

```
PDF ─▶ extract.py ─▶ clean.py ─▶ chunk.py ─▶ embeddings.py ─▶ store.py ─▶ ChromaDB
       (PyMuPDF)     (regex)     (heading-    (MiniLM)        (upsert)
                                  aware)
```

- **`extract.py`** — PyMuPDF reads each page with `sort=False` (the content
  stream is already in reading order; `sort=True` interleaves the columns).
  Tables are rendered to markdown via `find_tables()`.
- **`clean.py`** — strips headers/footers, un-glues mashed words, normalizes
  unicode/whitespace, smart title-cases.
- **`chunk.py`** — splits policy pages on numbered subsection headings
  (`5.1.`, `10.13.`), tolerating watermark artifacts and reflowed titles.
  Front-matter pages become whole-page chunks; oversized sections split on
  paragraph→sentence boundaries under a soft token cap. Each chunk has a
  **content-addressed id** so re-ingestion upserts instead of duplicating.
- **`store.py`** — embeds the path-prefixed `embedding_text` and upserts into a
  cosine ChromaDB collection.

### Phase 2 — Retrieval (`retrieval/`)

**Goal:** given a question, return the most relevant sections with a full score
breakdown. Hybrid search beats either method alone.

```
            ┌─ vector search (cosine, top 20) ─┐
query ──────┤                                  ├─ RRF fuse ─ cross-encoder rerank ─ post-process ─ results
            └─ BM25 search  (top 20) ──────────┘
```

1. **Vector** (`vector.py`) — semantic similarity, same MiniLM model as Phase 1.
2. **BM25** (`bm25.py`) — lexical match + a synonym map (`WFH → work from home`).
3. **Fusion** (`fusion.py`) — Reciprocal Rank Fusion (k=60) merges by *rank*,
   avoiding incompatible score scales.
4. **Rerank** (`rerank.py`) — cross-encoder rescores the fused top-20 (biggest
   precision win).
5. **Post-process** (`postprocess.py`) — cross-reference expansion, multi-part
   stitching, sibling hints.

Output is a typed `RetrievalResult` with all four scores, page provenance, an
`origin` tag, and a ready `citation`.

### Phase 3 — Multi-agent graph (`agents/`)

**Goal:** a conversational assistant that routes, rewrites, retrieves, answers
*only* from excerpts, self-grades, and remembers.

```
START → router ─(handbook)─→ rewriter → retriever → answer → grader ─(pass)─→ finalize → END
              └─(chitchat/    ↑                                  │
                 out_of_scope)│                                  └─(fail & retries left)─→ rewriter
                              └────────────→ respond ───────────────────────────────────→ finalize
```

| Node | LLM? | Job |
|------|:----:|-----|
| router | yes | classify turn → `handbook`/`chitchat`/`out_of_scope` |
| respond | yes | answer small talk / explain scope (no retrieval) |
| rewriter | yes | resolve coreferences, map vocab, split into ≤3 sub-queries; on retry, reformulate using the grader's notes |
| retriever | **no** | pure tool node: Phase 2 per sub-query, dedupe, rank |
| answer | yes | draft grounded only in excerpts, with inline citations |
| grader | yes | `{grounded, complete, reasoning}` → pass or retry |
| finalize | no | emit answer; if retries exhausted, an honest "couldn't confirm" + HR contact |

Key properties: **structured Pydantic outputs** (no string parsing), a real
**grader→rewriter retry loop** (trustworthiness over fluency), **per-thread
memory** via the SQLite checkpointer, and a **pluggable LLM provider**.

### Phase 3.5 — MCP (`mcp_server.py`)

Exposes only `search_handbook(query, k)` over stdio so any Cursor chat can query
the handbook. Cursor's model does the answering; the tool returns ranked
excerpts + `grounding_rules`. No API key.

### Phase 4 — Observability (`observability.py`)

A thin Langfuse wrapper, **no-op when unconfigured**. Wraps each turn in a
trace; nests every node/LLM/tool call under a LangChain callback handler; adds
custom spans with the per-chunk score breakdown; pushes scores
(`grader_grounded`, `grader_complete`, `retrieval_top_score`, `user_feedback`);
serves the answer prompt from Prompt Management with a hardcoded fallback.

### Phase 5 — Evaluation (`evaluation/`)

A hand-verified golden dataset (12 core questions, never LLM-rewritten) plus
synthetic questions for manual review; RAGAS metrics + the custom LLM-free
`section_hit_rate`; a chunking A/B experiment (heading-aware vs recursive);
and a **regression guard** (`run --check` exits nonzero on a >5% drop vs
`baseline.json`). Judge calls are cached by content hash.

### Phase 6 — API + UI (`api/`, `ui/`)

- **FastAPI backend**: `POST /ask` (SSE stream), `POST /feedback` (→ Langfuse),
  `GET /health` (503 if a critical component is down), `GET /sections`,
  `GET /threads/{id}` (proves memory). Async end-to-end via `AsyncSqliteSaver`.
  Auto-falls back to the `fake` provider if a hosted key is missing, so it
  always boots.
- **Streamlit UI**: live agent-activity line, streamed tokens, a **Sources**
  expander, a **How this answer was made** expander (node path, retries, grader
  verdict), 👍/👎 feedback, a clickable handbook index, and `thread_id`
  persistence for follow-ups.

---

## 4. End-to-end data flow (one question)

Asking *"How many earned leaves do I get per year?"* via the UI:

```
Browser (Streamlit)
   │  POST /ask  {question, thread_id}
   ▼
FastAPI  ──▶  HandbookAgent.astream_turn()         [Langfuse trace opens]
   │
   ├─ router    → "handbook"                         ─┐
   ├─ rewriter  → ["earned leave annual entitlement"] │  each step streamed
   ├─ retriever → Phase 2 hybrid search → top chunks  │  as an SSE `status`
   ├─ answer    → "You get 15 days… [5.1 …, p.24]"    │  event to the browser
   ├─ grader    → {grounded:true, complete:true} pass ─┘
   └─ finalize  → append to thread history
   │
   ▼  SSE: status* → token* → citations → trace → done
Browser renders answer + Sources + trace; 👍/👎 → POST /feedback → Langfuse score
```

Every box above is recorded as a nested span in Langfuse, with scores attached
to the trace, so a wrong answer can be debugged by seeing exactly what was
retrieved and how it was graded.

---

## 5. Cross-cutting design principles

- **ChromaDB is the single source of truth.** Vectors *and* chunk metadata live
  there; BM25 rebuilds from it. No second datastore to keep in sync.
- **Grounding over fluency.** The assistant cites everything, preserves units
  (calendar vs business days), and **refuses rather than guesses** on HR policy.
- **Runs with zero keys.** Local embeddings, local reranker, the `fake`
  provider, and offline `section_hit_rate` mean the full pipeline is testable in
  CI without spending a cent. Real LLMs (hosted or Ollama) are opt-in.
- **Pluggable everything.** LLM provider, embedding provider, and every
  retrieval/agent knob come from `.env` with defaults.
- **Graceful degradation.** Missing Langfuse keys, missing LLM keys, or an
  unavailable component degrade cleanly instead of crashing.
- **Typed contracts.** Pydantic for agent decisions and API requests; dataclasses
  for chunks and results; `TypedDict` for graph state.

---

## 6. Repository map

```
employee-handbook-rag/
├── ingestion/     Phase 1  PDF → chunks → ChromaDB
├── retrieval/     Phase 2  hybrid search (vector + BM25 + RRF + rerank)
├── agents/        Phase 3  LangGraph multi-agent assistant
├── mcp_server.py  Phase 3.5  Cursor MCP tool
├── observability.py  Phase 4  Langfuse tracing/scoring
├── evaluation/    Phase 5  RAGAS + section_hit_rate + chunking A/B
├── api/           Phase 6  FastAPI backend (SSE)
├── ui/            Phase 6  Streamlit chat UI
├── tests/         one test module per phase (69 tests)
├── docker-compose.yml, */Dockerfile
└── README.md · README-OBSERVABILITY.md · README-DEPLOY.md · ARCHITECTURE.md
```

See `README.md` for per-phase usage, `README-OBSERVABILITY.md` for Langfuse
setup, and `README-DEPLOY.md` for the deployment guide (local + Docker + Ollama).
