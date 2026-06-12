# Phase 4 — Observability with Langfuse

Every user turn is traced end-to-end in [Langfuse](https://langfuse.com): the
route decision, each node, every LLM call (with token cost), the per-sub-query
retrieval with full score breakdowns, and the grader's verdict. This is the
debugging surface for the whole project — when an answer is wrong, the trace
shows exactly what happened and why.

## Setup

1. Create a project at https://cloud.langfuse.com (or self-host) and copy the
   keys into `.env`:

   ```env
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
   ```

2. Run as usual — tracing turns on automatically:

   ```bash
   python -m agents.chat --thread demo1 --user-id alice
   ```

   With **no keys set**, the app logs one line (`Langfuse tracing DISABLED …`)
   and runs normally — tracing never blocks or crashes the app.

3. (Optional, for the prompt-management demo) In the Langfuse UI → **Prompts** →
   *New prompt*, name it **`handbook-answer-v1`**, type **Text**, and paste the
   answer system prompt (the text in `agents/prompts.py: ANSWER_SYSTEM`). The
   answer node fetches it (cached ~60s); edit it in the UI and the next answer
   uses the new text — no redeploy. If the prompt doesn't exist, the hardcoded
   `ANSWER_SYSTEM` is used as a fallback.

## What gets instrumented

| Mechanism | Covers |
|-----------|--------|
| **LangChain `CallbackHandler`** passed via `graph.invoke(config=…)` | Every node, LLM call (Anthropic/OpenAI model + **token usage/cost**), and tool call — auto-nested. No per-call hand-instrumentation. |
| **`turn_trace` span** (`observability.py`) | One trace per turn; `session_id = thread_id` (groups a conversation), `user_id` (CLI flag). |
| **Trace tags + metadata** | Tag = route (`handbook` / `chitchat` / `out_of_scope`); `+ "retried"` when the grader loop fired. Metadata: `retry_count`, `node_path` (e.g. `router>rewriter>retriever>answer>grader>finalize`), `rewritten_queries`. |
| **`retrieve_subquery` spans** | One per sub-query; metadata lists every chunk with `section_number`, `vector`, `bm25`, `fused`, `rerank` scores and `origin` (`hybrid` / `expanded_from:X` / `stitched:X`). |
| **`bm25_index_load` span** | The one-time BM25 index build (startup cost). |
| **Scores** | `grader_grounded` (0/1), `grader_complete` (0/1) with the grader's reasoning as the comment; `retrieval_top_score` (best rerank score); `user_feedback` (+1/-1 from the CLI). |

The trace tree for a handbook turn looks like:

```
handbook-turn                      ← trace root (tags: handbook [, retried])
├─ router            (LLM)         ← route decision
├─ rewriter          (LLM)         ← sub-queries
├─ retrieve_subquery (span)        ← chunks + vector/bm25/fused/rerank scores
│  └─ (bm25_index_load, first turn only)
├─ retrieve_subquery (span)        ← one per sub-query
├─ answer            (LLM)         ← uses prompt "handbook-answer-v1"
└─ grader            (LLM)         ← grounded/complete → scores on the trace
```

## Reading one trace top-to-bottom

> _Screenshot placeholder: `docs/langfuse-trace.png` — a handbook-turn trace
> with the span tree on the left and the selected span's metadata on the right._

1. **Tracing → Traces** (or **Sessions** → your `thread_id` to see the whole
   conversation). Open a trace named **`handbook-turn`**.
2. **Top of the trace:** the **route tag** (`handbook`/`chitchat`/`out_of_scope`),
   `user_id`, `session_id`, and the metadata block (`node_path`, `retry_count`,
   `rewritten_queries`).
3. **Per-sub-query retrieval:** expand each **`retrieve_subquery`** span and open
   its **Metadata** — the `chunks` array shows, for each fetched section, the
   `vector` / `bm25` / `fused` / `rerank` scores and its `origin`.
4. **Grader scores:** the **Scores** panel shows `grader_grounded`,
   `grader_complete` (hover for the reasoning comment), and `retrieval_top_score`.
5. **Token cost / latency:** each LLM observation (router/rewriter/answer/grader)
   shows model, input/output tokens, cost, and duration; the trace header shows
   totals.

## Debugging recipes (for this project)

**a. "The answer cited the wrong section."**
Open the trace → the relevant **`retrieve_subquery`** span → **Metadata → chunks**.
Compare `rerank` vs `bm25` vs `vector` for the wrong section vs the one you
expected: a high `bm25` but low `rerank` means a lexical false-positive the
reranker should have demoted; check `origin` — if it's `expanded_from:X` or
`stitched:X`, post-processing injected it (it wasn't a top hybrid hit), so the
answer node over-weighted a cross-referenced section.

**b. "Answer says 'not found' but the policy exists."**
Open the **`rewriter`** LLM observation → its **Output** (`rewritten_queries`).
Did the rewrite drop handbook vocabulary (e.g. turned "comp off" into something
that lost "compensatory off", or over-narrowed a multi-part question)? Then open
the matching **`retrieve_subquery`** span: if the right section isn't in `chunks`
at all, it's a rewrite/retrieval miss, not an answer-node problem. Fix by adding
a synonym (`retrieval/bm25.py: SYNONYMS`) or the rewriter vocabulary map.

**c. "Response was slow."**
In the trace, sort observations by **duration** (or scan the latency bars).
- A long **`bm25_index_load`** → first-turn startup cost only.
- A long **`retrieve_subquery`** → the cross-encoder reranker (CPU model load /
  scoring).
- Multiple **`answer`/`grader`** pairs and a `retried` tag → the grader bounced
  the draft and the loop re-ran (`retry_count` in metadata). That's LLM latency,
  not retrieval.

## Filtering the dashboard to bad answers

To find ungrounded answers (the highest-value filter):

- **Tracing → Traces → Filter →** add a **Scores** filter: `grader_grounded` **=
  0** (Numeric). That lists every turn where the grader judged the draft not
  fully supported by the excerpts.
- Combine with the **`retried`** tag to see cases the retry loop couldn't fix,
  or sort by `retrieval_top_score` ascending to correlate **low retrieval
  confidence** with bad answers.

## Graceful degradation & flushing

- Missing/invalid keys → one warning, tracing disabled, app runs normally.
- Any Langfuse error is swallowed (logged at debug) — it never breaks a turn.
- The CLI calls `observability.flush()` on exit so short-lived runs don't lose
  buffered traces.
