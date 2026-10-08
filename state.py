"""
state.py

Purpose & Use:
Defines the shared state schema (`AgentState`) that flows through all nodes and conditional
edges within the LangGraph text-to-SQL state machine.

Why this file exists:
- LangGraph requires a typed structure to manage state across nodes.
- Each node reads the current state and returns a dictionary of updated fields, which
  LangGraph merges into the workflow state.
- Centralizes the schema for tracking user queries, generated SQL, validation errors,
  execution results, retry counters, and synthesized final answers.

Fields:
- `question`: The user's natural language input.
- `schema_context`: Compact database schema text injected into the LLM prompt.
- `generated_sql`: The latest SQL query produced by the LLM.
- `validation_error`: Error message if query fails static security/syntax rules.
- `execution_result`: Structured execution result dictionary from the database.
- `retry_count`: Number of self-correction retry attempts completed.
- `max_retries`: Maximum allowed self-correction attempts before terminating.
- `final_answer`: Plain-English answer synthesized from the execution results.
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
