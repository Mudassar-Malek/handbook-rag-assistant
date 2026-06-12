"""Streamlit chat UI for the Employee Handbook assistant.

Talks ONLY to the FastAPI backend (API_BASE_URL, default http://localhost:8000).
Surfaces every Phase 1-5 feature: citations (Phase 1/2), live agent activity +
node trace (Phase 3), thread memory (Phase 3 checkpointer), and feedback ->
Langfuse (Phase 4).

Run:  streamlit run ui/app.py
"""

from __future__ import annotations

import os
import uuid

import streamlit as st

from client import APIError, ask_stream, get_health, get_sections, send_feedback

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Employee Handbook Assistant", page_icon="📘", layout="centered")


# --------------------------------------------------------------------------- state
def _init_state():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: role, content, citations, trace, trace_id
    if "pending" not in st.session_state:
        st.session_state.pending = None


def _new_conversation():
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.messages = []
    st.session_state.pending = None


# ------------------------------------------------------------------------- sidebar
def render_sidebar():
    with st.sidebar:
        st.header("📘 Handbook Assistant")
        if st.button("➕ New conversation", use_container_width=True):
            _new_conversation()
            st.rerun()

        st.caption(f"Thread: `{st.session_state.thread_id[:8]}…`")

        # API health badge.
        try:
            health = get_health(API_BASE_URL)
            ok = health.get("status") == "ok"
            st.success("API: healthy") if ok else st.warning(f"API: {health.get('status')}")
        except APIError:
            st.error("API: offline")

        st.divider()
        st.subheader("Handbook index")
        try:
            sections = get_sections(API_BASE_URL)
            for sec in sections:
                label = f"{sec['section_number']}  {sec['section_title']}"
                if st.button(label, key=f"sec-{sec['section_number']}", use_container_width=True):
                    st.session_state.pending = (
                        f"What does section {sec['section_number']} cover?"
                    )
                    st.rerun()
        except APIError as e:
            st.info(f"Section index unavailable.\n\n{e}")

        st.divider()
        st.caption(
            "Answers are generated from the 2026 Employee Handbook. For binding "
            "policy decisions, contact HR."
        )


# --------------------------------------------------------------------- rendering
def render_history():
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_sources(msg.get("citations") or [])
                _render_trace(msg.get("trace"))
                _render_feedback(msg.get("trace_id"), key=i)


def _render_sources(citations: list[dict]):
    if not citations:
        return
    with st.expander(f"📎 Sources ({len(citations)})"):
        for c in citations:
            pages = ", ".join(str(p) for p in c.get("pages", []))
            st.markdown(
                f"**[{c['section_number']} {c['section_title']}]** — p.{pages}\n\n"
                f"> {c.get('snippet', '')}"
            )


def _render_trace(trace: dict | None):
    if not trace:
        return
    with st.expander("🛠 How this answer was made"):
        st.markdown(f"**Node path:** `{trace.get('node_path', '')}`")
        st.markdown(f"**Retries:** {trace.get('retry_count', 0)}")
        grader = trace.get("grader")
        if grader:
            st.markdown(
                f"**Grader:** grounded={grader.get('grounded')} · complete={grader.get('complete')}"
            )
        if trace.get("trace_id"):
            st.caption(f"Langfuse trace: `{trace['trace_id']}`")


def _render_feedback(trace_id: str | None, key: int):
    cols = st.columns([1, 1, 8])
    up = cols[0].button("👍", key=f"up-{key}")
    down = cols[1].button("👎", key=f"down-{key}")
    if up or down:
        if not trace_id:
            cols[2].info("Feedback needs Langfuse enabled (set LANGFUSE_* in .env).")
            return
        try:
            send_feedback(API_BASE_URL, trace_id, 1 if up else -1)
            cols[2].success("Thanks for the feedback!")
        except APIError as e:
            cols[2].error(str(e))


# ----------------------------------------------------------------------- one turn
def run_turn(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        activity = st.empty()
        answer_box = st.empty()
        steps: list[str] = []
        tokens: list[str] = []
        citations: list[dict] = []
        trace: dict | None = None
        thread_id = st.session_state.thread_id

        try:
            for ev in ask_stream(API_BASE_URL, question, thread_id, user_id="ui"):
                etype = ev.get("type")
                if etype == "status":
                    node, detail = ev.get("node"), ev.get("detail")
                    if node == "start":
                        thread_id = detail or thread_id
                        st.session_state.thread_id = thread_id
                        continue
                    steps.append(f"{node} ({detail})" if detail else node)
                    activity.caption("🧠 Agent activity: " + "  →  ".join(steps))
                elif etype == "token":
                    tokens.append(ev.get("content", ""))
                    answer_box.markdown("".join(tokens) + "▌")
                elif etype == "citations":
                    citations = ev.get("items", [])
                elif etype == "trace":
                    trace = ev
                elif etype == "error":
                    answer_box.error(f"Error: {ev.get('message')}")
                    return
                elif etype == "done":
                    thread_id = ev.get("thread_id", thread_id)
                    st.session_state.thread_id = thread_id
        except APIError as e:
            answer_box.error(
                f"Couldn't reach the assistant API.\n\n{e}\n\n"
                f"Is the backend running at `{API_BASE_URL}`?"
            )
            return

        final = "".join(tokens).strip() or "_(no answer)_"
        answer_box.markdown(final)
        _render_sources(citations)
        _render_trace(trace)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final,
        "citations": citations,
        "trace": trace,
        "trace_id": (trace or {}).get("trace_id"),
    })
    st.rerun()  # re-render so the per-message feedback buttons get stable keys


# --------------------------------------------------------------------------- main
def main():
    _init_state()
    render_sidebar()
    st.title("Employee Handbook Assistant")
    st.caption("Ask about leave, attendance, conduct, benefits, IT/security, or the exit process.")

    render_history()

    typed = st.chat_input("Ask a question about the handbook…")
    pending = st.session_state.pending
    st.session_state.pending = None
    question = typed or pending
    if question:
        run_turn(question)


if __name__ == "__main__":
    main()
