"""Phase 3 multi-agent system (LangGraph) for the Employee Handbook RAG project.

A StateGraph wires together five cooperating "agents" (nodes):

    router -> rewriter -> retriever -> answer -> grader

with conditional edges for routing (chitchat / out-of-scope skip retrieval) and
for a grounded-answer retry loop. A SQLite checkpointer gives per-thread memory
so follow-up questions work across turns.

Entry point: `python -m agents.chat --thread demo1`.
"""

__version__ = "0.1.0"
