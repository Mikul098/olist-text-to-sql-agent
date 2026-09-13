"""
execute_sql.py

Runs a SQL query that has ALREADY passed validate_sql() against the database,
and returns a structured result so the LangGraph node can decide what to do
next (succeed -> synthesize_response, fail -> loop back to generate_sql).

This function does NOT call validate_sql() itself -- that's a deliberate
separation of concerns. The LangGraph node calls validate_sql() first, and
only calls this if validation passed. Keeping them separate makes each one
easier to test on its own.

Two safety nets live here specifically because this hits a real database:
  1. A hard row cap, so a query that accidentally matches all 100k+ orders
     doesn't blow up your context window when it gets handed to the LLM
     for response synthesis.
  2. A query timeout, so a slow/expensive query (e.g. an accidental cross
     join) doesn't hang the whole Streamlit app.
"""

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from db_connection import get_engine

MAX_ROWS = 200          # hard cap on rows returned to the LLM
QUERY_TIMEOUT_SECONDS = 15


def execute_sql(sql: str) -> dict:
    """
    Executes a validated SELECT query.

    Returns a dict shaped like:
      {
        "success": bool,
        "rows": list[dict] | None,      # None on failure
        "row_count": int | None,        # actual matched row count before capping
        "truncated": bool,              # True if row_count > MAX_ROWS
        "columns": list[str] | None,
        "error": str | None,            # populated only on failure
      }
    """
    engine = get_engine()

    try:
        with engine.connect() as conn:
            # MySQL-side timeout so a runaway query can't hang the app.
            conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={QUERY_TIMEOUT_SECONDS * 1000}"))

            result = conn.execute(text(sql))
            columns = list(result.keys())
            all_rows = result.fetchall()

        row_count = len(all_rows)
        truncated = row_count > MAX_ROWS
        limited_rows = all_rows[:MAX_ROWS]

        rows_as_dicts = [dict(zip(columns, row)) for row in limited_rows]

        return {
            "success": True,
            "rows": rows_as_dicts,
            "row_count": row_count,
            "truncated": truncated,
            "columns": columns,
            "error": None,
        }

    except SQLAlchemyError as e:
        # SQLAlchemy wraps the real MySQL error -- surface the useful part.
        error_message = str(e.orig) if hasattr(e, "orig") and e.orig else str(e)
        return {
            "success": False,
            "rows": None,
            "row_count": None,
            "truncated": False,
            "columns": None,
            "error": error_message,
        }

    except Exception as e:
        # Catch-all so an unexpected error still returns a structured result
        # instead of crashing the LangGraph node.
        return {
            "success": False,
            "rows": None,
            "row_count": None,
            "truncated": False,
            "columns": None,
            "error": f"Unexpected error: {e}",
        }


if __name__ == "__main__":
    # Quick manual test -- run `python execute_sql.py` to confirm real queries
    # work end-to-end against your actual database.
    test_queries = [
        "SELECT * FROM customers LIMIT 3;",
        "SELECT customer_city, COUNT(*) AS total FROM customers GROUP BY customer_city ORDER BY total DESC LIMIT 5;",
        "SELECT * FROM this_table_does_not_exist;",  # should fail cleanly
    ]

    for sql in test_queries:
        print(f"Query: {sql}")
        result = execute_sql(sql)
        if result["success"]:
            print(f"  Success. row_count={result['row_count']} truncated={result['truncated']}")
            print(f"  Columns: {result['columns']}")
            print(f"  First row: {result['rows'][0] if result['rows'] else None}")
        else:
            print(f"  Failed. Error: {result['error']}")
        print()
