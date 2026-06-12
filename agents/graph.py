"""The LangGraph StateGraph: nodes, conditional edges, and the checkpointer.

Graph shape:

    START -> router -> (handbook) -> rewriter -> retriever -> answer -> grader
                    -> (else)     -> respond -------------------------------> finalize -> END
                                                                grader -(pass)-> finalize
                                                                grader -(retry)-> rewriter
"""

from __future__ import annotations

import re
import sqlite3

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

import observability as obs

from .config import AgentSettings, get_agent_settings
from .llm import build_llm
from .prompts import ANSWER_SYSTEM
from .schemas import GradeOutput
from .state import AgentState, fresh_turn_state, result_to_excerpt


class HandbookAgent:
    """Builds and runs the multi-agent handbook assistant."""

    def __init__(self, settings: AgentSettings | None = None, llm=None, retriever=None,
                 checkpointer=None):
        self.settings = settings or get_agent_settings()
        # Initialize tracing first (no-op if LANGFUSE_* are unset).
        obs.init_observability()
        self.llm = llm or build_llm(self.settings)
        self._retriever = retriever  # built lazily (loads the reranker model)
        # The API injects an AsyncSqliteSaver so the whole stack is async; the
        # CLI/tests fall back to the sync SqliteSaver. Either satisfies the
        # checkpointer protocol the compiled graph needs.
        self._checkpointer = checkpointer or SqliteSaver(
            sqlite3.connect(self.settings.checkpoint_db, check_same_thread=False)
        )
        self.graph = self._build()
        self.last_trace_id: str | None = None

    # The Phase 2 retriever is heavy (cross-encoder); only build it when needed.
    @property
    def retriever(self):
        if self._retriever is None:
            from retrieval.config import get_settings as get_retrieval_settings
            from retrieval.retriever import Retriever

            self._retriever = Retriever(get_retrieval_settings(top_k=self.settings.retrieval_k))
        return self._retriever

    # ------------------------------------------------------------------ nodes
    def router_node(self, state: AgentState) -> dict:
        """Classify the latest turn: handbook vs chitchat vs out_of_scope.

        Uses conversation history so follow-ups ("and sick leave?") are routed
        to the handbook lane rather than mis-classified as small talk.
        """
        decision = self.llm.route(state["messages"], state["original_query"])
        return {"route": decision.route, "trace": state.get("trace", []) + [f"router({decision.route})"]}

    def respond_node(self, state: AgentState) -> dict:
        """Answer chitchat / out_of_scope directly, with no retrieval."""
        text = self.llm.smalltalk(state["route"], state["messages"], state["original_query"])
        return {"final_answer": text, "trace": state["trace"] + ["respond"]}

    def rewriter_node(self, state: AgentState) -> dict:
        """Turn the question into 1-3 standalone, handbook-vocabulary queries.

        Resolves follow-up coreferences, maps casual terms, and decomposes
        multi-part questions. On a retry it produces a DIFFERENT formulation,
        guided by the grader's notes about what was missing.
        """
        grade = state.get("grade")
        retry = state.get("retry_count", 0)
        missing = ""
        is_retry = bool(grade) and not (grade["grounded"] and grade["complete"])
        if is_retry:
            retry += 1
            missing = grade["reasoning"]

        out = self.llm.rewrite(
            state["messages"], state["original_query"], state.get("previous_rewrites", []), missing
        )
        subs = out.rewritten_queries[:3]
        return {
            "rewritten_queries": subs,
            "previous_rewrites": state.get("previous_rewrites", []) + subs,
            "retry_count": retry,
            "grade": None,  # clear so finalize doesn't misread a stale grade
            "trace": state["trace"] + ["rewriter"],
        }

    def retriever_node(self, state: AgentState) -> dict:
        """Pure tool node: retrieve for each sub-query, dedupe, rank by rerank.

        No LLM call here. Union of all sub-query hits, deduplicated by chunk id,
        ordered by best rerank score, capped at max_context_chunks.

        Each sub-query is wrapped in its own Langfuse span carrying the full
        score breakdown per chunk — the single most important debugging view:
        when an answer is wrong, this shows exactly what was fetched and why.
        """
        k = self.settings.retrieval_k
        best: dict[str, tuple[float, object]] = {}
        for sub in state["rewritten_queries"]:
            with obs.span("retrieve_subquery", input={"sub_query": sub, "k": k}) as sp:
                results = self.retriever.retrieve(sub)
                sp.update(
                    metadata={
                        "sub_query": sub,
                        "k": k,
                        "n_results": len(results),
                        "chunks": [
                            {
                                "section_number": r.section_number,
                                "section_title": r.section_title,
                                "vector": r.scores.vector,
                                "bm25": r.scores.bm25,
                                "fused": r.scores.fused,
                                "rerank": r.scores.rerank,
                                # "hybrid" | "expanded_from:X" | "stitched:X"
                                "origin": r.origin,
                            }
                            for r in results
                        ],
                    },
                    output=[r.section_number for r in results],
                )
            for r in results:
                score = r.scores.rerank if r.scores.rerank is not None else float("-inf")
                if r.id not in best or score > best[r.id][0]:
                    best[r.id] = (score, r)
        ordered = sorted(best.values(), key=lambda t: t[0], reverse=True)
        excerpts = [result_to_excerpt(r) for _, r in ordered[: self.settings.max_context_chunks]]
        return {
            "retrieval": excerpts,
            "trace": state["trace"] + [f"retriever({len(state['rewritten_queries'])} queries)"],
        }

    def answer_node(self, state: AgentState) -> dict:
        """Draft an answer grounded ONLY in the retrieved excerpts, with citations.

        The system prompt is fetched from Langfuse Prompt Management
        ("handbook-answer-v1"), cached, with the hardcoded ANSWER_SYSTEM as
        fallback — so it can be edited in the UI without a redeploy.
        """
        system = obs.get_answer_system_prompt(ANSWER_SYSTEM)
        draft = self.llm.answer(state["original_query"], state["retrieval"], system=system)
        return {"draft_answer": draft, "trace": state["trace"] + ["answer"]}

    def grader_node(self, state: AgentState) -> dict:
        """Check the draft is grounded (numbers!) and complete; structured output."""
        grade: GradeOutput = self.llm.grade(
            state["original_query"], state["draft_answer"], state["retrieval"]
        )
        tag = "pass" if grade.passed else "fail"
        return {"grade": grade.model_dump(), "trace": state["trace"] + [f"grader({tag})"]}

    def finalize_node(self, state: AgentState) -> dict:
        """Produce the final answer and append it to the conversation history.

        - chitchat/out_of_scope: use the direct response.
        - graded pass: use the draft.
        - retries exhausted: an honest "couldn't fully answer" (never a forced
          answer), plus the closest sections and the HR contact.
        """
        route = state.get("route")
        if route in ("chitchat", "out_of_scope"):
            final = state["final_answer"]
        else:
            grade = state.get("grade")
            if grade and grade["grounded"] and grade["complete"]:
                final = state["draft_answer"]
            else:
                final = self._honest_incomplete(state)
        return {
            "final_answer": final,
            "messages": [AIMessage(content=final)],
            "trace": state["trace"] + ["finalize"],
        }

    def _honest_incomplete(self, state: AgentState) -> str:
        cites = ", ".join(ex["citation"] for ex in state.get("retrieval", [])[:4])
        note = (state.get("grade") or {}).get("reasoning", "")
        msg = (
            "I wasn't able to fully confirm an answer to this from the handbook, so I "
            "don't want to guess on an HR policy."
        )
        if note:
            msg += f" (Gap: {note})"
        if cites:
            msg += f" The closest sections are {cites}."
        msg += f" Please confirm with HR at {self.settings.hr_email}."
        return msg

    # ------------------------------------------------------------ conditional edges
    def _route_gate(self, state: AgentState) -> str:
        return "rewriter" if state["route"] == "handbook" else "respond"

    def _grade_gate(self, state: AgentState) -> str:
        grade = state["grade"]
        if grade["grounded"] and grade["complete"]:
            return "finalize"
        if state.get("retry_count", 0) < self.settings.max_retries:
            return "rewriter"
        return "finalize"

    # ------------------------------------------------------------------ build/run
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("router", self.router_node)
        g.add_node("respond", self.respond_node)
        g.add_node("rewriter", self.rewriter_node)
        g.add_node("retriever", self.retriever_node)
        g.add_node("answer", self.answer_node)
        g.add_node("grader", self.grader_node)
        g.add_node("finalize", self.finalize_node)

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
        return g.compile(checkpointer=self._checkpointer)

    def chat(self, query: str, thread_id: str, user_id: str = "dev") -> AgentState:
        """Run one turn in the given thread; returns the final state.

        Wraps the whole turn in one Langfuse trace (session_id = thread_id), and
        passes the callback handler through invoke config so every node / LLM /
        tool call nests under it automatically.
        """
        handler = obs.get_callback_handler()
        with obs.turn_trace(
            name="handbook-turn", session_id=thread_id, user_id=user_id, input_text=query
        ):
            config: dict = {"configurable": {"thread_id": thread_id}}
            if handler is not None:
                config["callbacks"] = [handler]
            state = self.graph.invoke(fresh_turn_state(query), config=config)
            self._emit_trace_observability(state)
            self.last_trace_id = obs.current_trace_id()
        return state

    def _emit_trace_observability(self, state: AgentState) -> None:
        """Set trace tags/metadata and push scores from the finished turn."""
        route = state.get("route", "")
        retry_count = state.get("retry_count", 0)
        # Node path with the (…) annotations stripped, e.g.
        # "router>rewriter>retriever>answer>grader>finalize".
        node_path = ">".join(re.sub(r"\(.*?\)", "", t) for t in state.get("trace", []))

        tags = [route] if route else []
        if retry_count and retry_count > 0:
            tags.append("retried")

        obs.update_current_trace(
            output=state.get("final_answer", ""),
            tags=tags,
            metadata={
                "route": route,
                "retry_count": retry_count,
                "node_path": node_path,
                "n_sub_queries": len(state.get("rewritten_queries", [])),
                "rewritten_queries": state.get("rewritten_queries", []),
            },
        )

        # Grader scores (handbook route only).
        grade = state.get("grade")
        if grade:
            reason = grade.get("reasoning", "")
            obs.score_current_trace("grader_grounded", 1.0 if grade.get("grounded") else 0.0,
                                    comment=reason)
            obs.score_current_trace("grader_complete", 1.0 if grade.get("complete") else 0.0,
                                    comment=reason)

        # Retrieval confidence = best rerank score across retrieved excerpts.
        rerank_scores = [
            ex.get("rerank") for ex in state.get("retrieval", []) if ex.get("rerank") is not None
        ]
        if rerank_scores:
            obs.score_current_trace("retrieval_top_score", float(max(rerank_scores)))

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    # ------------------------------------------------------------ async streaming
    def _status_event(self, node: str, delta: dict) -> dict | None:
        """Translate one streamed node update into a typed `status` event.

        These are the visible payoff of the multi-agent design: the UI can show
        what each agent is doing while the user waits.
        """
        if node == "router":
            return {"type": "status", "node": "router", "detail": delta.get("route", "")}
        if node == "rewriter":
            n = len(delta.get("rewritten_queries", []))
            return {"type": "status", "node": "rewriter", "detail": f"{n} sub-queries"}
        if node == "retriever":
            n = len(delta.get("retrieval", []))
            return {"type": "status", "node": "retriever", "detail": f"{n} chunks"}
        if node == "answer":
            return {"type": "status", "node": "answer", "detail": "drafting answer"}
        if node == "grader":
            g = delta.get("grade") or {}
            verdict = "pass" if (g.get("grounded") and g.get("complete")) else "retry"
            return {"type": "status", "node": "grader", "detail": verdict}
        if node == "respond":
            return {"type": "status", "node": "respond", "detail": "direct reply"}
        return None  # finalize emits no status; `done` closes the stream

    async def astream_turn(self, query: str, thread_id: str, user_id: str = "anon"):
        """Async generator of typed events for one turn (drives the SSE endpoint).

        Order: status* -> token* -> citations -> trace -> done. Wrapped in one
        Langfuse trace (session_id = thread_id) exactly like the sync `chat()`.
        """
        handler = obs.get_callback_handler()
        with obs.turn_trace(
            name="handbook-turn", session_id=thread_id, user_id=user_id, input_text=query
        ):
            config: dict = {"configurable": {"thread_id": thread_id}}
            if handler is not None:
                config["callbacks"] = [handler]

            async for chunk in self.graph.astream(
                fresh_turn_state(query), config=config, stream_mode="updates"
            ):
                for node, delta in chunk.items():
                    ev = self._status_event(node, delta or {})
                    if ev is not None:
                        yield ev

            # Authoritative end-of-turn state (handles retries cleanly).
            snap = await self.graph.aget_state(config)
            state = snap.values if snap else {}

            # Stream the produced answer as token events (provider-agnostic).
            answer = state.get("final_answer", "") or ""
            for piece in _stream_pieces(answer):
                yield {"type": "token", "content": piece}

            yield {
                "type": "citations",
                "items": [
                    {
                        "section_number": ex["section_number"],
                        "section_title": ex["section_title"],
                        "pages": ex["pages"],
                        "snippet": (ex.get("text") or "")[:200],
                    }
                    for ex in state.get("retrieval", [])
                ],
            }

            # Trace-level tags/metadata/scores, then surface the trace id.
            self._emit_trace_observability(state)
            self.last_trace_id = obs.current_trace_id()

            grade = state.get("grade") or {}
            node_path = ">".join(re.sub(r"\(.*?\)", "", t) for t in state.get("trace", []))
            yield {
                "type": "trace",
                "node_path": node_path,
                "grader": (
                    {"grounded": bool(grade.get("grounded")), "complete": bool(grade.get("complete"))}
                    if grade else None
                ),
                "retry_count": state.get("retry_count", 0),
                "trace_id": self.last_trace_id,
            }
            yield {"type": "done", "thread_id": thread_id}


def _stream_pieces(text: str):
    """Split an answer into word-sized pieces (keeps trailing spaces) so the UI
    can render it token-by-token."""
    if not text:
        return
    for match in re.findall(r"\S+\s*", text):
        yield match
