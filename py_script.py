"""
py_script.py

Purpose & Use:
Data ingestion and ETL script responsible for loading raw Olist Brazilian e-commerce
CSV datasets from the `data/` directory into the MySQL database (`olist_db`).

What this file does:
1. Connects to the local MySQL instance via SQLAlchemy using PyMySQL.
2. Loads dataset CSV files from the `data/` directory into their respective MySQL tables in a Foreign-Key-safe order:
   - Independent parent tables: `customers`, `product_category_name_translation`, `products`
   - Dependent parent table: `orders` (depends on `customers`)
   - Dependent child tables: `order_items`, `order_payments`, `order_reviews` (depend on `orders` & `products`)
3. Streams data in manageable batches (chunksize = 10,000 rows) to prevent high memory consumption.
4. Sanitizes Pandas `NaN` values to `None` (SQL NULL) so typed MySQL columns (INT, DATETIME)
   insert cleanly without conversion errors.

When to use:
Run once during initial database setup to populate the MySQL tables with the raw dataset records:
    python py_script.py
"""

from pathlib import Path
import pandas as pd
import os
import urllib.parse
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Directory containing the CSV datasets
DATA_DIR = Path(__file__).resolve().parent / "data"

user = os.getenv("DB_USER", "root")
raw_password = os.getenv("DB_PASSWORD", "")
host = os.getenv("DB_HOST", "localhost")
port = os.getenv("DB_PORT", "3306")
db_name = os.getenv("DB_NAME", "olist_db")
password = urllib.parse.quote_plus(raw_password)

# 1. Connect to the local MySQL server
engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}')

# 2. Define files and target tables, IN FK-SAFE ORDER
datasets = {
    'olist_customers_dataset.csv': 'customers',
    'product_category_name_translation.csv': 'product_category_name_translation',
    'olist_products_dataset.csv': 'products',
    'olist_orders_dataset.csv': 'orders',
    'olist_order_items_dataset.csv': 'order_items',
    'olist_order_payments_dataset.csv': 'order_payments',
    'olist_order_reviews_dataset.csv': 'order_reviews',
}

# 3. Import data in manageable chunks
chunksize = 10000

for file_name, table_name in datasets.items():
    file_path = DATA_DIR / file_name

    # Check if table is already populated to avoid Primary Key duplicate errors
    with engine.connect() as conn:
        existing_rows = conn.execute(text(f"SELECT COUNT(1) FROM {table_name}")).scalar()
    
    if existing_rows > 0:
        print(f"Table '{table_name}' already contains {existing_rows:,} rows. Skipping.")
        continue

    print(f"Loading {file_name} into {table_name} table...")

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # Replace NaN with None so MySQL INT/DATETIME columns don't choke
        # on pandas' float NaN representation of missing values.
        chunk = chunk.replace({np.nan: None})

        chunk.to_sql(name=table_name, con=engine, if_exists='append', index=False)

    print(f"Successfully loaded all data into {table_name}.\n")

print("\nAll tables verified and ready.")