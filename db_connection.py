"""
Central place to create the SQLAlchemy engine for olist_db.
Every other module (schema context, SQL execution node, etc.)
should import get_engine() from here instead of building its own connection.
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
    engine = get_engine()
    with engine.connect() as conn:
        print("Connected successfully to:", engine.url.database)
