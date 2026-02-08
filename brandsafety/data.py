import os
import numpy as np
import pandas as pd

def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".parquet", ".pq"]:
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                "Failed to read parquet. Install parquet extra: pip install 'brandsafety-svm[parquet]' "
                "or 'pip install -e .[parquet]' from repo."
            ) from e
    raise ValueError(f"Unsupported file extension: {ext}. Use .csv or .parquet")

def extract_embeddings(df, *, embedding_prefix = "emb_", embedding_col = None):
    if embedding_col is not None:
        if embedding_col not in df.columns:
            raise ValueError(f"embedding_col='{embedding_col}' not found in df")
        arrs = df[embedding_col].values
        X = np.vstack([np.asarray(a, dtype = np.float32) for a in arrs])
        return X

    if embedding_prefix is None:
        raise ValueError("Provide embedding_prefix (wide cols) or embedding_col (list/array col).")

    cols = [c for c in df.columns if str(c).startswith(embedding_prefix)]
    if not cols:
        raise ValueError(f"No embedding columns found with prefix='{embedding_prefix}'")

    def _col_key(c):
        tail = str(c)[len(embedding_prefix):]
        return int(tail) if tail.isdigit() else tail

    cols_sorted = sorted(cols, key = _col_key)
    X = df[cols_sorted].values.astype(np.float32, copy = False)
    return X
