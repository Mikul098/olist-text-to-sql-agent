"""
state.py

The shared state object that flows through every node in the graph.
Each node reads from this and returns a dict of the fields it updates --
LangGraph merges that into the running state automatically.
"""

from typing import TypedDict, Optional


class AgentState(TypedDict):
    question: str                      # the user's original natural language question
    schema_context: str                # full schema text, computed once per app session

    generated_sql: Optional[str]       # most recent SQL the LLM produced
    validation_error: Optional[str]    # set if validate_sql() rejected the query
    execution_result: Optional[dict]   # the dict returned by execute_sql()

    retry_count: int                   # how many correction attempts made so far
    max_retries: int                   # ceiling before we give up and report failure

    final_answer: Optional[str]        # natural language answer, set by synthesize_response
