"""
Generates a compact, LLM-friendly description of the olist_db schema.

Why this exists:
Handing the LLM raw CREATE TABLE statements is fine, but SQL generation
quality improves noticeably when the model also sees a few real sample
rows per table -- it picks up on actual value formats (e.g. order_status
values like 'delivered'/'shipped', category strings like 'beleza_saude',
date formats, etc.) instead of guessing.

This module is meant to be called ONCE per app session (schema doesn't
change at runtime), and the resulting string gets cached and injected
into the SQL-generation prompt in the LangGraph node.
"""

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from db_connection import get_engine

# Tables to include, in a sensible reading order for the LLM.
# (Order doesn't affect correctness, just readability of the prompt.)
TABLE_ORDER = [
    "customers",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "products",
    "product_category_name_translation",
]

SAMPLE_ROWS_PER_TABLE = 3


def _get_foreign_keys(inspector, table_name: str) -> list[str]:
    """Returns formatted FK descriptions declared in the schema."""
    fks = inspector.get_foreign_keys(table_name)
    descriptions = []
    for fk in fks:
        local_cols = ", ".join(fk["constrained_columns"])
        ref_table = fk["referred_table"]
        ref_cols = ", ".join(fk["referred_columns"])
        descriptions.append(f"{table_name}.{local_cols} -> {ref_table}.{ref_cols}")
    return descriptions


def _get_sample_rows(engine: Engine, table_name: str, limit: int) -> str:
    """Fetches a few sample rows and formats them as a readable block."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
        rows = result.fetchall()
        columns = result.keys()

    if not rows:
        return "  (no sample rows available)"

    lines = []
    for row in rows:
        row_dict = dict(zip(columns, row))
        lines.append(f"  {row_dict}")
    return "\n".join(lines)


def build_schema_context(engine: Engine | None = None) -> str:
    """
    Builds the full schema context string:
      - table name
      - each column with its type
      - declared foreign keys
      - sample rows

    Returns a single string ready to drop into an LLM prompt.
    """
    if engine is None:
        engine = get_engine()

    inspector = inspect(engine)
    all_tables = inspector.get_table_names()

    sections = []
    all_fk_lines = []

    for table_name in TABLE_ORDER:
        if table_name not in all_tables:
            continue  # skip gracefully if a table is missing

        columns = inspector.get_columns(table_name)
        pk_constraint = inspector.get_pk_constraint(table_name)
        pk_cols = set(pk_constraint.get("constrained_columns", []))

        column_lines = []
        for col in columns:
            col_name = col["name"]
            col_type = str(col["type"])
            marker = " [PK]" if col_name in pk_cols else ""
            column_lines.append(f"  - {col_name}: {col_type}{marker}")

        fk_lines = _get_foreign_keys(inspector, table_name)
        all_fk_lines.extend(fk_lines)

        sample_block = _get_sample_rows(engine, table_name, SAMPLE_ROWS_PER_TABLE)

        section = (
            f"TABLE: {table_name}\n"
            f"Columns:\n" + "\n".join(column_lines) + "\n"
            f"Sample rows:\n{sample_block}"
        )
        sections.append(section)

    fk_summary = "\n".join(f"  - {line}" for line in all_fk_lines) if all_fk_lines else "  (none declared)"

    # Note: product_category_name in `products` and `product_category_name_translation`
    # are related by matching string values, not a declared FK -- worth telling the LLM
    # explicitly since the inspector won't surface it.
    notes = (
        "IMPORTANT NOTES:\n"
        "  - products.product_category_name and product_category_name_translation.product_category_name "
        "are related by matching text values (not a declared foreign key) -- join on this column directly "
        "when you need the English category name.\n"
        "  - order_reviews has no formal primary key and may contain duplicate review_id values.\n"
        "  - Only SELECT statements are permitted against this database."
    )

    context = (
        "DATABASE SCHEMA: olist_db (Brazilian E-Commerce dataset)\n\n"
        + "\n\n".join(sections)
        + "\n\nDECLARED FOREIGN KEYS:\n"
        + fk_summary
        + "\n\n"
        + notes
    )

    return context


if __name__ == "__main__":
    # Run `python schema_context.py` to preview exactly what gets sent to the LLM.
    print(build_schema_context())
