"""
synthesize_response.py

Purpose & Use:
Response synthesis node. Converts raw tabular SQL query results into clear,
plain-English, conversational answers for the user.

What this file does:
1. `synthesize_response()`:
   - Takes the user's natural language question and the execution result dictionary
     produced by `execute_sql.py`.
   - Prompts the LLM (temperature=0.3) to answer concisely using ONLY the provided data,
     avoiding technical SQL/database jargon and never inventing values.
   - Transparently indicates if the returned rows were truncated by the row cap.
   - Handles empty query results cleanly without confusing the user.
2. `build_failure_message()`:
   - Formulates a helpful, transparent explanation when the agent exhausts its retry
     budget (`max_retries`) without a successful query, outlining the last error and
     suggesting how the user can rephrase.

Where it is used:
- Invoked by `synthesize_response_node` (success path) and `give_up_node` (exhausted retries)
  in `graph.py`.

Testing:
Run directly to test response synthesis against mock query rows:
    python synthesize_response.py
"""

from llm_client import get_llm_client, MODEL_NAME

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a Brazilian e-commerce dataset.
You will be given the user's original question and the query result (as rows of data).
Write a clear, concise, natural-language answer using ONLY the data provided.
Do not make up numbers that aren't in the data. If the data is empty, say so plainly.
Do not mention SQL, tables, or databases in your answer -- just answer the question naturally,
the way you'd explain it to someone who doesn't know what a database is.
"""


def synthesize_response(question: str, execution_result: dict) -> str:
    """
    execution_result is the dict returned by execute_sql():
      {"success": True, "rows": [...], "row_count": int, "truncated": bool, "columns": [...]}
    """
    if not execution_result["rows"]:
        return "I ran the query, but it didn't return any results for that question."

    client = get_llm_client()

    rows_preview = execution_result["rows"]
    truncated_note = ""
    if execution_result["truncated"]:
        truncated_note = (
            f"\n\n(Note: the full result had {execution_result['row_count']} rows; "
            f"only the first {len(rows_preview)} are shown here.)"
        )

    user_message = (
        f"Question: {question}\n\n"
        f"Query result data:\n{rows_preview}"
        f"{truncated_note}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,  # a little more natural phrasing is fine here, unlike SQL generation
    )

    return response.choices[0].message.content.strip()


def build_failure_message(question: str, last_error: str) -> str:
    """Used when the retry loop exhausts max_retries without a successful query."""
    return (
        f"I wasn't able to find an answer to \"{question}\" after a few attempts. "
        f"The last issue was: {last_error}\n\n"
        f"Try rephrasing the question, or being more specific about what you're looking for."
    )


if __name__ == "__main__":
    # Manual test -- run `python synthesize_response.py` to see a real answer
    # generated from a fake-but-realistic result set (no DB call needed here,
    # this node only cares about the result dict shape, not where it came from).
    fake_result = {
        "success": True,
        "rows": [
            {"customer_state": "SP", "total": 41746},
            {"customer_state": "RJ", "total": 12852},
            {"customer_state": "MG", "total": 11635},
        ],
        "row_count": 3,
        "truncated": False,
        "columns": ["customer_state", "total"],
    }

    question = "Which states have the most customers?"
    answer = synthesize_response(question, fake_result)
    print("Answer:")
    print(answer)
