"""
generate_sql.py

The actual "Text-to-SQL" step. Takes:
  - the user's natural language question
  - the schema context (from schema_context.py)
  - OPTIONALLY: the previous attempt's SQL + error message, if this is a
    retry after validate_sql/execute_sql rejected something

...and returns a single, clean SQL string with no markdown fences,
explanations, or commentary -- just SQL, ready to hand to validate_sql().
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