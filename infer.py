import argparse
import os
import numpy as np

from brandsafety.data import load_table, extract_embeddings
from brandsafety.artifacts import load_artifact

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pkl artifact produced by train.py")
    ap.add_argument("--data", required=True, help="Input embeddings file (.parquet or .csv)")
    ap.add_argument("--embedding-prefix", default=None, help="Prefix for wide embedding columns (e.g., emb_)")
    ap.add_argument("--embedding-col", default=None, help="Single column containing embedding arrays")
    ap.add_argument("--out", required=True, help="Output scored file (.parquet or .csv)")
    ap.add_argument("--keep-cols", default="", help="Comma-separated columns to keep from input (default: keep all)")
    args = ap.parse_args()

    artifact = load_artifact(args.model)
    df = load_table(args.data)

    X = extract_embeddings(df, embedding_prefix = args.embedding_prefix, embedding_col = args.embedding_col)

    task = artifact.get("task")
    out_df = df.copy()

    if task == "binary":
        model = artifact["model"]
        p_hat = model.predict_proba(X)[:, 1]
        out_df["p_hat"] = p_hat

        thr = artifact.get("threshold", None)
        if thr is not None:
            out_df["yhat"] = (p_hat >= float(thr)).astype(int)

    elif task == "multiclass_ovr":
        classes = artifact["classes"]
        models = artifact["models"]

        probs = {}
        for c in classes:
            if c in models:
                probs[c] = models[c].predict_proba(X)[:, 1]
            else:
                probs[c] = np.full(shape = (X.shape[0],), fill_value = -1.0, dtype = float)

        for c in classes:
            out_df[f"prob_{c}"] = probs[c]

        M = np.vstack([probs[c] for c in classes]).T
        pred_idx = np.argmax(M, axis = 1)
        out_df["pred_label"] = [classes[i] for i in pred_idx]

    else:
        raise ValueError(f"Unknown task in artifact: {task}")

    keep = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
    if keep:
        pred_cols = [c for c in out_df.columns if c.startswith("prob_") or c in ["p_hat", "yhat", "pred_label"]]
        cols = [c for c in keep if c in out_df.columns] + pred_cols
        # de-dup preserving order
        seen = set()
        cols2 = []
        for c in cols:
            if c not in seen:
                cols2.append(c)
                seen.add(c)
        out_df = out_df[cols2]

    ext = os.path.splitext(args.out)[1].lower()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if ext in [".parquet", ".pq"]:
        out_df.to_parquet(args.out, index = False)
    elif ext == ".csv":
        out_df.to_csv(args.out, index = False)
    else:
        raise ValueError("Output must be .parquet or .csv")

    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
