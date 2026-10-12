"""
execute_sql.py

Purpose & Use:
Database execution engine for validated SQL queries. Runs read-only queries against
MySQL (`olist_db`) and returns a structured dictionary result for downstream agent nodes.

Why this file exists:
- Provides controlled database execution separated from SQL validation (`validate_sql.py`),
  maintaining clean modular separation of concerns.
- Implements runtime safety nets to protect application responsiveness and LLM context windows:
  1. Hard row cap (`MAX_ROWS = 200`): Prevents accidental massive result sets from
     overflowing the LLM's token context during response synthesis.
  2. Query timeout (`QUERY_TIMEOUT_SECONDS = 15`): Enforces `MAX_EXECUTION_TIME` in MySQL
     so accidental Cartesian products or runaway queries cannot hang the application.
  3. Structured error handling: Catches and unpacks `SQLAlchemyError` / MySQL error messages
     so the LangGraph workflow can feed them back to `generate_sql.py` for self-correction.

Return Shape:
{
    "success": bool,
    "rows": list[dict] | None,     # List of column-value mappings (None on error)
    "row_count": int | None,       # Total rows matched before row capping
    "truncated": bool,             # True if matching rows exceeded MAX_ROWS
    "columns": list[str] | None,   # List of result column names
    "error": str | None,           # Descriptive error message on failure
}

Where it is used:
- Called by `execute_sql_node` in `graph.py`.

Testing:
Run directly to execute sample queries against the active database:
    python execute_sql.py
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
            # MySQL-side timeout so a runaway query can't hang the app (if supported by server).
            try:
                conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={QUERY_TIMEOUT_SECONDS * 1000}"))
            except Exception:
                pass

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
