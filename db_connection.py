"""
db_connection.py

Purpose & Use:
Centralized database connection factory for the Olist Text-to-SQL system.
Manages a shared, reusable SQLAlchemy Engine connected to MySQL (`olist_db`).

Why this file exists:
- Provides a singleton database engine via `get_engine()` to avoid redundant connection pools.
- Decouples database connection configuration and credential handling from individual modules.
- Automatically loads credentials from the local `.env` file (with URL-encoded passwords for safety).
- Utilizes `pool_pre_ping=True` to proactively detect and recover from dropped/stale connections.

Where it is used:
- `schema_context.py`: Introspects table schemas, column types, foreign keys, and sample rows.
- `execute_sql.py`: Executes validated SELECT queries against the live database.

Testing:
Run directly to perform a connection sanity check:
    python db_connection.py
"""

import os
import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local ".env" file if present

_engine: Engine | None = None


def get_engine() -> Engine:
    """Returns a singleton SQLAlchemy engine, creating it on first call."""
    global _engine

    if _engine is not None:
        return _engine

    user = os.getenv("DB_USER", "root")
    raw_password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "olist_db")

    password = urllib.parse.quote_plus(raw_password)

    connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"

    _engine = create_engine(connection_string, pool_pre_ping=True)
    return _engine


if __name__ == "__main__":
    # Quick sanity check: run `python db_connection.py` to confirm the connection works.
    print(f"Testing MySQL connection to {os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'olist_db')}...")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            print("Connected successfully to:", engine.url.database)
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure MySQL service is running locally on port 3306.")
        print("2. Check credentials (DB_USER, DB_PASSWORD, DB_NAME) in your .env file.")
        print("3. Ensure database 'olist_db' is created (run 'python py_script.py').")
