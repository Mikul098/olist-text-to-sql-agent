"""
generate_sql.py

Purpose & Use:
The core Text-to-SQL generation node. Translates user natural language questions
into executable MySQL queries targeting `olist_db` using an LLM.

What this file does:
1. Takes the user's question and compact schema context (from `schema_context.py`).
2. Enforces strict generation rules in the system prompt:
   - Output ONLY a single SQL SELECT statement (no write/DDL operations).
   - Use only tables and columns from the provided schema context.
   - Include sensible LIMIT clauses unless aggregating.
   - Correctly join product tables to English category translations.
3. Cleans LLM output by removing markdown code fences (` ```sql ... ``` `).
4. Supports Self-Correction:
   - If a previous query failed validation or execution, feeds the failed SQL and
     error message back into the prompt so the model can fix the specific issue.

Where it is used:
- Called by `generate_sql_node` in `graph.py` on the initial attempt and on any retry loop.

Testing:
Run directly to test SQL generation against live schema context:
    python generate_sql.py
"""

import re
from llm_client import get_llm_client, MODEL_NAME

SYSTEM_PROMPT = """You are an expert MySQL query writer for a Brazilian e-commerce database (olist_db).

Rules you MUST follow:
- Output ONLY a single SQL SELECT statement. No explanations, no markdown code fences, no commentary.
- Only generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or any other write/DDL operation.
- Use only the tables and columns provided in the schema below. Do not invent column or table names.
- Always add a reasonable LIMIT (e.g. LIMIT 100) unless the question clearly asks for an aggregate
  (like a COUNT, AVG, or SUM) that naturally returns few rows.
- When joining products to their English category name, join products.product_category_name to
  product_category_name_translation.product_category_name (these are matched by text value, not a formal foreign key).

SCHEMA:
{schema_context}
"""


def _strip_markdown_fences(text: str) -> str:
    """Removes ```sql ... ``` or ``` ... ``` wrapping if the model adds it despite instructions."""
    text = text.strip()
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate_sql(
    question: str,
    schema_context: str,
    previous_sql: str | None = None,
    previous_error: str | None = None,
) -> str:
    """
    Generates a SQL query for the given natural language question.

    If previous_sql/previous_error are provided, this is treated as a
    self-correction retry -- the model is shown what it tried before and
    why it failed, so it can fix the actual mistake instead of guessing blind.
    """
    client = get_llm_client()

    system_message = SYSTEM_PROMPT.format(schema_context=schema_context)

    user_message = f"Question: {question}"

    if previous_sql and previous_error:
        user_message += (
            f"\n\nYour previous attempt failed. Fix the specific problem below -- "
            f"do not just repeat the same query.\n"
            f"Previous SQL:\n{previous_sql}\n\n"
            f"Error returned by the database:\n{previous_error}"
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0,  # deterministic-ish output is preferred for SQL generation
    )

    raw_output = response.choices[0].message.content
    return _strip_markdown_fences(raw_output)


if __name__ == "__main__":
    # Manual test -- run `python generate_sql.py` to see a real generated query,
    # using your ACTUAL database schema (pulled fresh from MySQL every time
    # this script runs -- in the real app we'll cache this instead, see note below).
    from schema_context import build_schema_context

    real_schema = build_schema_context()

    question = "How many customers are there in each state?"
    sql = generate_sql(question, real_schema)
    print("Generated SQL:")
    print(sql)