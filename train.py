import argparse
import json
import os
import numpy as np

from brandsafety.data import load_table, extract_embeddings
from brandsafety.split import split_by_bucket
from brandsafety.binary import train_binary
from brandsafety.multiclass import train_multiclass_ovr
from brandsafety.artifacts import save_artifact

def parse_list(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input labeled embeddings file (.parquet or .csv)")
    ap.add_argument("--mode", required=True, choices=["binary", "multiclass"], help="Training mode")

    ap.add_argument("--label-col", default="label")
    ap.add_argument("--bucket-col", default="sample_bucket")
    ap.add_argument("--anchor-value", default="anchor_unbiased")
    ap.add_argument("--train-values", default="mined_keywords,diversity_long_tail")

    ap.add_argument("--anchor-split", default="stratified", choices=["stratified", "group_target_root"])
    ap.add_argument("--group-col", default="target_root")
    ap.add_argument("--anchor-cal-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--embedding-prefix", default=None, help="Prefix for wide embedding columns (e.g., emb_)")
    ap.add_argument("--embedding-col", default=None, help="Single column containing embedding arrays")

    ap.add_argument("--cal-method", default="isotonic", choices=["isotonic", "sigmoid"])
    ap.add_argument("--target-precision", type=float, default=0.80, help="Binary precision target; also used for optional per-class thresholds")
    ap.add_argument("--prefer", default="max_recall", choices=["max_recall", "min_threshold", "max_f1"])

    ap.add_argument("--positive", default=None, help="Comma-separated positive class names for binary mode (e.g., explicit,partial)")

    ap.add_argument("--category", default="category", help="Metadata only")
    ap.add_argument("--out", required=True, help="Output .pkl path")
    ap.add_argument("--metrics-out", default=None, help="Optional metrics json path")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    df = load_table(args.data)

    df_train, df_cal, df_test = split_by_bucket(
        df,
        label_col = args.label_col,
        bucket_col = args.bucket_col,
        anchor_value = args.anchor_value,
        train_values = tuple(parse_list(args.train_values)),
        anchor_cal_frac = args.anchor_cal_frac,
        seed = args.seed,
        anchor_split = args.anchor_split,
        group_col = args.group_col
    )

    X_train = extract_embeddings(df_train, embedding_prefix = args.embedding_prefix, embedding_col = args.embedding_col)
    X_cal   = extract_embeddings(df_cal,   embedding_prefix = args.embedding_prefix, embedding_col = args.embedding_col)
    X_test  = extract_embeddings(df_test,  embedding_prefix = args.embedding_prefix, embedding_col = args.embedding_col)

    y_train = df_train[args.label_col].astype(str).values
    y_cal   = df_cal[args.label_col].astype(str).values
    y_test  = df_test[args.label_col].astype(str).values

    artifact = {
        "category": args.category,
        "meta": {
            "mode": args.mode,
            "label_col": args.label_col,
            "bucket_col": args.bucket_col,
            "anchor_value": args.anchor_value,
            "train_values": parse_list(args.train_values),
            "anchor_split": args.anchor_split,
            "group_col": args.group_col,
            "anchor_cal_frac": float(args.anchor_cal_frac),
            "seed": int(args.seed),
            "cal_method": args.cal_method,
            "target_precision": float(args.target_precision),
            "prefer": args.prefer,
            "feature_spec": {"embedding_prefix": args.embedding_prefix, "embedding_col": args.embedding_col},
            "sizes": {"train": int(len(df_train)), "cal": int(len(df_cal)), "test": int(len(df_test))},
        }
    }

    if args.mode == "binary":
        pos = set(parse_list(args.positive))
        if not pos:
            raise SystemExit("--positive is required for binary mode (e.g., --positive explicit or --positive partial,explicit)")

        ytr = np.isin(y_train, list(pos)).astype(int)
        yca = np.isin(y_cal,   list(pos)).astype(int)
        yte = np.isin(y_test,  list(pos)).astype(int)

        model, thr, metrics = train_binary(
            X_train, ytr,
            X_cal, yca,
            X_test, yte,
            cal_method = args.cal_method,
            target_precision = args.target_precision,
            prefer = args.prefer,
            seed = args.seed,
            verbose = args.verbose
        )

        artifact.update({
            "task": "binary",
            "classes": ["neg", "pos"],
            "positive_classes": sorted(list(pos)),
            "model": model,
            "threshold": float(thr),
            "metrics": metrics
        })

    else:
        classes = sorted(list(set(y_train) | set(y_cal) | set(y_test)))
        models, thresholds, metrics = train_multiclass_ovr(
            X_train, y_train,
            X_cal, y_cal,
            X_test, y_test,
            classes = classes,
            cal_method = args.cal_method,
            target_precision = args.target_precision if args.target_precision is not None else None,
            prefer = args.prefer,
            seed = args.seed,
            verbose = args.verbose
        )

        artifact.update({
            "task": "multiclass_ovr",
            "classes": classes,
            "models": models,
            "thresholds": thresholds,
            "metrics": metrics
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_artifact(artifact, args.out)

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(artifact.get("metrics", {}), f, indent = 2)

    print(f"Saved artifact: {args.out}")
    if args.metrics_out:
        print(f"Saved metrics: {args.metrics_out}")

if __name__ == "__main__":
    main()
