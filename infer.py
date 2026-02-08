"""Optional CLI wrapper around the library API."""

import argparse
import os
from brandsafety.api import load_artifact
from brandsafety.data import load_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embedding-prefix", default="emb_")
    ap.add_argument("--embedding-col", default=None)
    args = ap.parse_args()

    artifact = load_artifact(args.model)
    df = load_table(args.data)
    scored = artifact.predict_df(df, embedding_prefix = args.embedding_prefix, embedding_col = args.embedding_col)

    ext = os.path.splitext(args.out)[1].lower()
    if ext in [".csv"]:
        scored.to_csv(args.out, index = False)
    elif ext in [".parquet", ".pq"]:
        scored.to_parquet(args.out, index = False)
    else:
        raise ValueError("out must be .csv or .parquet")

    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
