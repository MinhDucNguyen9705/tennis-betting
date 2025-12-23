import duckdb
import pandas as pd
from config import PARQUET_GLOB, DUCKDB_THREADS

con = duckdb.connect(database=":memory:")

con.execute(f"""
CREATE OR REPLACE VIEW matches AS
SELECT * FROM read_parquet('{PARQUET_GLOB}');
""")
con.execute(f"PRAGMA threads={DUCKDB_THREADS};")

def sql_df(q: str, params: dict | None = None) -> pd.DataFrame:
    return con.execute(q, params or {}).df()

def existing_cols() -> set[str]:
    return set(sql_df("DESCRIBE matches")["column_name"].tolist())