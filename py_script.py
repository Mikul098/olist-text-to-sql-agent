import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import numpy as np

# Safely encode your password just in case it has numbers/symbols that conflict
password = urllib.parse.quote_plus("Ankita@14")

# 1. Connect to the local MySQL server
engine = create_engine(f'mysql+pymysql://root:{password}@localhost:3306/olist_db')

# 2. Define files and target tables, IN FK-SAFE ORDER
# Rules:
#   - customers has no dependencies -> load first
#   - product_category_name_translation has no dependencies
#   - products has no dependencies (category name is just a string column, not an FK here)
#   - orders depends on customers -> must come after customers
#   - order_items depends on orders AND products -> must come after both
#   - order_payments depends on orders
#   - order_reviews depends on orders
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
    print(f"Loading {file_name} into {table_name} table...")

    for chunk in pd.read_csv(file_name, chunksize=chunksize):
        # Replace NaN with None so MySQL INT/DATETIME columns don't choke
        # on pandas' float NaN representation of missing values.
        chunk = chunk.replace({np.nan: None})

        chunk.to_sql(name=table_name, con=engine, if_exists='append', index=False)

    print(f"Successfully loaded all data into {table_name}.\n")

print("All tables loaded successfully.")