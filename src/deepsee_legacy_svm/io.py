import os
import pandas as pd

def load_table(path: str, **kwargs) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, **kwargs)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Unsupported file type: {ext}")
