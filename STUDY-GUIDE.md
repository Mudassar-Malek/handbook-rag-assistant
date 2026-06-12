# Study Guide — Explaining This Project (Interview-Ready)

A complete, code-backed walkthrough so you can confidently answer **any** question
about this RAG system. Read top to bottom once; revisit Parts A, B, and D before
an interview.

**Contents**
- [Part A — The 30-second pitch](#part-a--the-30-second-pitch)
- [Part B — One question, traced through the code](#part-b--one-question-traced-through-the-code)
- [Part C — Each layer, with its key code](#part-c--each-layer-with-its-key-code)
- [Part D — Q&A bank (the questions you'll actually get)](#part-d--qa-bank-the-questions-youll-actually-get)
- [Part E — The "why" cheat sheet](#part-e--the-why-cheat-sheet)
- [Glossary](#glossary)

---

## Part A — The 30-second pitch

> "It's a RAG assistant that answers HR-policy questions from a handbook with exact
> citations. It's six layers: **ingestion** (PDF → heading-aware chunks → ChromaDB),
> **hybrid retrieval** (vector + BM25 + Reciprocal Rank Fusion + a cross-encoder
> reranker), a **LangGraph multi-agent** system (router → rewriter → retriever →
> answer → a self-grading retry loop), **Langfuse** observability, a **RAGAS**
> evaluation harness with a regression gate, and a **FastAPI + Streamlit** app.
> It's provider-pluggable — Anthropic, OpenAI, or local Ollama — and the whole
> thing runs offline with zero API keys."

**The one-sentence version:** "A production-grade RAG assistant with a self-grading
multi-agent pipeline that cites its sources and refuses to hallucinate HR policy."

---

## Part B — One question, traced through the code

When a user asks *"How many earned leaves do I get per year?"*, here is the actual path.

### 1. Request enters the API — `api/main.py`

```python
@app.post("/ask")
@limiter.limit("10/minute")
async def ask(request: Request, body: AskRequest):
    agent: HandbookAgent = request.app.state.agent
    thread_id = body.thread_id or uuid.uuid4().hex

    async def event_source():
        yield {"data": json.dumps({"type": "status", "node": "start", "detail": thread_id})}
        try:
            async for ev in agent.astream_turn(body.question, thread_id, body.user_id):
                yield {"data": json.dumps(ev)}
```

### 2. The graph runs node by node — `agents/graph.py`

```python
g.add_edge(START, "router")
g.add_conditional_edges("router", self._route_gate,
                        {"rewriter": "rewriter", "respond": "respond"})
g.add_edge("respond", "finalize")
g.add_edge("rewriter", "retriever")
g.add_edge("retriever", "answer")
g.add_edge("answer", "grader")
g.add_conditional_edges("grader", self._grade_gate,
                        {"finalize": "finalize", "rewriter": "rewriter"})
g.add_edge("finalize", END)
```

- **router** classifies → `"handbook"`, so `_route_gate` sends it to **rewriter**.
- **rewriter** turns it into a clean search query, e.g. `["earned leave annual entitlement days"]`.
- **retriever** (no LLM) calls the Phase 2 retriever per sub-query.
- **answer** drafts the response grounded in the excerpts.
- **grader** returns `{grounded, complete}`; `_grade_gate` decides pass→finalize or fail→rewriter (retry).

### 3. The grader is the trust mechanism — `agents/graph.py`

```python
def _grade_gate(self, state: AgentState) -> str:
    grade = state["grade"]
    if grade["grounded"] and grade["complete"]:
        return "finalize"
    if state.get("retry_count", 0) < self.settings.max_retries:
        return "rewriter"
    return "finalize"
```

### 4. Events stream back

The agent yields `status → token → citations → trace → done`, which the Streamlit UI
renders live (agent activity line, streamed answer, Sources expander, trace expander).

**That's the whole loop.** Everything in Part C is "how each box works."

---

## Part C — Each layer, with its key code

### Phase 1 — Ingestion: *why heading-aware chunking?*

The core decision: split on the handbook's own structure, not arbitrary character
counts. The chunker starts a new chunk at each numbered subsection heading.

```python
# ingestion/chunk.py
HEADING_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.?\s+(\S.*)$")
```

The pipeline is three steps — extract, clean, chunk:

```python
# ingestion/ingest.py
def build_chunks(pdf_path: str, sort: bool):
    raw_pages = extract_pages(pdf_path, sort=sort)
    cleaned = {p.number: clean_page(p.text) for p in raw_pages}
    source = Path(pdf_path).name
    ...
```

Each chunk also gets a **content-addressed id** (`source-section-part-hash`) so
re-ingesting *upserts* instead of duplicating, and a separate `embedding_text`
(`"Leave Policies > 5.1 Earned Leave (EL):\n<body>"`) — the path-prefixed string
that actually gets embedded, while `body` stays as clean display text.

**Talking point:** "Each chunk = one policy subsection (e.g. `5.1 Earned Leave`),
so a retrieved chunk is a complete, citable answer unit. I proved this beats naive
splitting in Phase 5 — 1.00 vs 0.58 page-hit-rate."

### Phase 2 — Hybrid retrieval: *why two searches + fusion + rerank?*

The whole pipeline is one readable method:

```python
# retrieval/retriever.py
def retrieve(self, query: str) -> list[RetrievalResult]:
    s = self.settings
    # 1. Two independent ranked lists.
    vector_hits = self.vector.search(query, s.vector_n)
    bm25_hits = self.bm25.search(query, s.bm25_n)
    # 2. Fuse with RRF.
    fused = reciprocal_rank_fusion([vector_hits, bm25_hits], k=s.rrf_k)
    fused_ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    candidate_ids = [cid for cid, _ in fused_ranked[: s.fetch_k]]
    # 3. Rerank the fused top-N, keep top_k.  4. Post-process. (below)
```

Why each piece (the most-asked area):

- **Vector search** = *meaning*. Good for paraphrases ("time off" → "leave"). Chroma
  returns cosine *distance*; we convert to similarity (`1 - distance`) so higher = better.
- **BM25** = *exact keywords*. Catches acronyms embeddings miss (`EL`, `LOP`, `POSH`).
  It expands query synonyms first (`WFH → work from home teleworking`).
- **RRF** merges the two using only *rank*, sidestepping incompatible score scales:

```python
# retrieval/fusion.py
def reciprocal_rank_fusion(ranked_lists, k=60) -> dict[str, float]:
    fused = {}
    for hits in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(hits, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused
```

- **Cross-encoder reranker** = the precision win. It reads the query and candidate
  *together* (not as separate vectors) and scores relevance directly — expensive,
  so it only runs on the fused top-20.
- **Post-processing** (after rerank): cross-reference expansion ("see Section 5.11"),
  multi-part stitching (fetch sibling parts of split sections), and sibling hints.

### Phase 3 — The agent: *why a graph, not a single prompt?*

The state is a typed dict threaded through every node:

```python
# agents/state.py
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # persisted per thread_id
    original_query: str
    route: str                    # "handbook" | "chitchat" | "out_of_scope"
    rewritten_queries: list[str]
    previous_rewrites: list[str]  # for retries
    retrieval: list[dict]
    draft_answer: str
    grade: Optional[dict]
    retry_count: int
    final_answer: str
    trace: list[str]
```

Every agent decision is a **Pydantic schema** — the model is forced to return typed,
validated fields, not free text:

```python
# agents/schemas.py
class GradeOutput(BaseModel):
    grounded: bool = Field(description="True only if EVERY factual claim "
                           "(especially numbers/dates/durations) is supported.")
    complete: bool = Field(description="True only if all parts were addressed.")
    reasoning: str = Field(description="What is missing (used to guide a retry).")
```

The nodes at a glance:

| Node | LLM? | Job |
|------|:----:|-----|
| router | yes | classify the turn → handbook / chitchat / out_of_scope |
| respond | yes | answer small talk / explain scope (no retrieval) |
| rewriter | yes | resolve coreferences, map vocab, decompose into ≤3 sub-queries; on retry, reformulate using the grader's notes |
| retriever | **no** | pure tool node: Phase 2 search per sub-query, dedupe, rank |
| answer | yes | draft grounded ONLY in excerpts, with inline citations |
| grader | yes | `{grounded, complete, reasoning}` → pass or retry |
| finalize | no | emit the answer; if retries exhausted, honest "couldn't confirm" + HR contact |

**Talking point:** "A single prompt can't *retry*. The graph lets the grader bounce
a weakly-grounded answer back to the rewriter with notes on what was missing — so
for HR facts it refuses rather than guesses."

### Provider abstraction — *why it runs with no key*

One factory builds whatever provider you configured:

```python
# agents/llm.py
def build_llm(settings: AgentSettings) -> LLMClient:
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model=settings.anthropic_model, temperature=settings.temperature)
        return LangChainLLM(model, settings)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model = ChatOllama(model=settings.ollama_model, temperature=settings.temperature,
                           base_url=settings.ollama_base_url)
        return LangChainLLM(model, settings, structured_method="json_schema")
    if provider == "fake":
        from .fake_llm import FakeLLM
        return FakeLLM(settings)
    ...
```

`fake` is a deterministic, rule-based double implementing the same interface, so the
whole graph runs offline in CI. Ollama uses JSON-schema structured outputs so the
router/grader still return valid Pydantic on local models.

### Phase 4 — Observability: *how do you debug a wrong answer?*

The clever bit — tracing **never crashes the app**. If keys are missing, every call
is a no-op:

```python
# observability.py
@contextlib.contextmanager
def turn_trace(name, session_id, user_id, input_text):
    c = _state["client"]
    if not _state["enabled"] or c is None:
        yield _NoopSpan()
        return
    ...
```

When enabled, each turn is one Langfuse trace (session = thread_id); a LangChain
callback handler auto-nests every node/LLM/tool call; the retriever adds a span per
sub-query with the full per-chunk score breakdown; and scores (grader grounded/
complete, retrieval confidence, user feedback) are pushed to the trace.

### Phase 5 — Evaluation: *how do you know it's good?*

The headline metric needs no LLM and is dead simple — did retrieval surface the
section that actually contains the answer?

```python
# evaluation/metrics.py
def section_hit(gt_sections, retrieved_sections) -> bool:
    """True iff every ground-truth section was retrieved (gt ⊆ retrieved)."""
    return set(gt_sections) <= set(retrieved_sections)

def section_hit_rate(rows) -> float:
    if not rows:
        return 0.0
    return sum(section_hit(gt, got) for gt, got in rows) / len(rows)
```

Plus: RAGAS for `faithfulness` / `answer_relevancy` / `answer_correctness` (full
pipeline) and `context_precision` / `context_recall` (retrieval); a hand-verified
golden dataset (12 core questions an LLM never rewrites); judge-call caching; and a
**regression gate** — `run --check` exits nonzero if any aggregate drops >5% below
`baseline.json` (CI-ready).

### Phase 6 — API + UI + deployment

- **FastAPI** loads the heavy models (retriever, BM25, cross-encoder, compiled graph)
  exactly once via lifespan; `AsyncSqliteSaver` makes the stack async end-to-end.
- **`/ask`** streams SSE (`status → token → citations → trace → done`), rate-limited.
- **Streamlit** UI talks only to the API; shows live agent activity, streamed tokens,
  a Sources expander, a "How this answer was made" trace expander, and feedback.
- **Docker** multi-stage builds with model-cache volumes; `docker-compose` wires
  the `api` + `ui` services.

---

## Part D — Q&A bank (the questions you'll actually get)

**Q: What is RAG and why use it here?**
> Retrieval-Augmented Generation: instead of relying on the LLM's memory, you retrieve
> relevant document chunks and feed them into the prompt so answers are grounded in
> real source text. Essential for HR policy where hallucination has real consequences —
> and it means answers cite exact sections.

**Q: Why hybrid search instead of just embeddings?**
> Embeddings capture meaning but miss exact short tokens people type — acronyms like
> `EL`, `POSH`, `LOP`. BM25 nails those lexically. Fusing both with RRF gets the best
> of semantic + keyword. The cross-encoder reranker then re-scores the top candidates
> by reading query+chunk together for final precision.

**Q: What does RRF actually do / why k=60?**
> It merges ranked lists using only each item's rank: each list contributes
> `1/(k+rank)`. That avoids comparing a cosine similarity (0–1) against a BM25 score
> (unbounded). k=60 is the standard constant; it dampens the influence of top ranks
> so one list can't dominate.

**Q: Why LangGraph and not a simple chain?**
> Because I needed a control loop: a grader that can *send work back*. A linear chain
> can't conditionally retry. The graph models router/rewriter/retriever/answer/grader
> as nodes with conditional edges, and a typed state object carries everything between
> them.

**Q: How does it avoid hallucinating?**
> Three layers: the answer prompt says "only from these excerpts, else say you couldn't
> find it," the grader independently checks every number is supported by the excerpts,
> and if it fails after retries, `finalize` returns an honest "couldn't confirm — contact
> HR" instead of a forced answer.

**Q: How does memory / follow-ups work?**
> A SQLite checkpointer (`langgraph-checkpoint-sqlite`) persists conversation state
> keyed by `thread_id`. The `messages` field uses the `add_messages` reducer so history
> accumulates. So "and for adoption leave?" is routed as a handbook follow-up and the
> rewriter resolves it against the prior turn.

**Q: How is the cross-encoder different from the embedding model?**
> The embedding (bi-encoder) encodes query and doc *separately* into vectors then
> compares — fast, scalable, approximate. The cross-encoder feeds query+doc *together*
> through the model and outputs one relevance score — slower but much more accurate, so
> I only run it on ~20 fused candidates.

**Q: What happens with no API key?**
> Embeddings and the reranker are local models, the `fake` provider runs the graph
> deterministically, and `section_hit_rate` needs no LLM. So the full pipeline and test
> suite run free and offline. For real fluent answers you set a provider key or run
> Ollama locally.

**Q: How do you know your chunking choice was right?**
> I ran an A/B: my heading-aware chunks (Strategy A) vs naive
> `RecursiveCharacterTextSplitter` (Strategy B) on the same retriever. A scored 1.00
> page-hit-rate vs B's 0.58 — B's 1000-token chunks swallowed multiple leave
> subsections, so a "sick leave" query pulled a generic blob. Data-backed decision.

**Q: How would you scale this to many documents / production?**
> Swap ChromaDB for a hosted vector DB (pgvector/Pinecone), move embeddings/reranker to
> a GPU service, add per-document metadata filtering, cache retrievals, and put the
> FastAPI behind a load balancer. The observability + eval harness already give you the
> production feedback loop.

**Q: What are the weaknesses / what would you improve?**
> It's single-document and tuned to this handbook's structure. The `/ask` streaming is
> pseudo-streaming (answer generated then chunked). And evaluation depends on a small
> hand-built golden set. Next steps: true token streaming, a larger eval set, and
> generalizing the chunker.

**Q: Why ChromaDB?**
> It's a local, embedded, persistent vector DB — no server to run, cosine space out of
> the box. It's the single source of truth: vectors *and* chunk metadata live there, so
> BM25 rebuilds its index from the same data instead of a second datastore.

**Q: What are the structured outputs for?**
> Every agent decision (route, rewrite, grade) is a Pydantic model returned via the
> LLM's structured-output mode. No fragile string parsing — the model must return
> typed, validated fields, which makes the graph's branching reliable.

**Q: How is observability wired without slowing things down?**
> A LangChain callback handler is passed through the graph's invoke config, so every
> node/LLM/tool call auto-nests under one trace — no manual per-call instrumentation.
> And it degrades to a no-op when Langfuse keys aren't set, so it never blocks or
> crashes the app.

**Q: What's the MCP server for?**
> It exposes the *retriever* (not the full agent) as a Cursor tool, `search_handbook`.
> Cursor's own model is the LLM, so it just returns ranked excerpts + grounding rules —
> no API key needed. Lets you query the handbook from any Cursor chat.

---

## Part E — The "why" cheat sheet

| Decision | Why |
|---|---|
| Heading-aware chunking | Chunks = complete citable policy units; proven > naive splitting |
| Hybrid vector + BM25 | Semantic recall + exact-acronym precision |
| RRF fusion | Combines lists with incompatible score scales |
| Cross-encoder rerank | Highest precision, applied only to top candidates |
| LangGraph state machine | Enables the grader→rewriter retry loop |
| Pydantic structured outputs | No fragile string parsing; validated decisions |
| Grader node | Refuses / retries rather than hallucinating HR facts |
| ChromaDB as single source of truth | Vectors + metadata in one place; BM25 rebuilds from it |
| Pluggable providers + `fake` | Runs offline, free, CI-friendly |
| Langfuse no-op when unconfigured | Tracing never crashes the app |
| `section_hit_rate` | One interpretable, LLM-free quality number |
| Content-addressed chunk ids | Re-ingestion upserts, never duplicates |

**If you can speak to Part B (the trace), the hybrid-retrieval reasoning, and the
grader retry loop, you'll handle ~90% of questions confidently.**

---

## Glossary

- **RAG** — Retrieval-Augmented Generation: retrieve relevant text, then generate an
  answer grounded in it.
- **Embedding** — a vector representation of text; similar meanings → nearby vectors.
- **Bi-encoder** — encodes query and document separately (fast, approximate); used for
  vector search.
- **Cross-encoder** — encodes query+document together (slow, accurate); used for reranking.
- **BM25** — a classic lexical ranking function based on term frequency; great for exact
  keywords/acronyms.
- **RRF (Reciprocal Rank Fusion)** — merges ranked lists by summing `1/(k+rank)`.
- **Chunk** — a unit of document text that gets embedded and retrieved; here, one policy
  subsection.
- **ChromaDB** — a local embedded vector database.
- **LangGraph** — a library for building LLM apps as state machines (nodes + edges).
- **Checkpointer** — persists graph state per `thread_id` (this is the memory).
- **Structured output** — forcing the LLM to return a validated, typed object (Pydantic).
- **Grounding** — ensuring every claim in the answer is supported by retrieved text.
- **Langfuse** — an LLM observability platform (traces, scores, prompt management).
- **RAGAS** — a library of metrics for evaluating RAG systems.
- **SSE (Server-Sent Events)** — a one-way HTTP stream; used to push answer tokens to the UI.
- **MCP (Model Context Protocol)** — a standard for exposing tools to AI agents (e.g. Cursor).

---

*See also: `README.md` (per-phase usage), `ARCHITECTURE.md` (system architecture +
tech stack), `README-OBSERVABILITY.md` (Langfuse), `README-DEPLOY.md` (deployment).*
