"""Optional CLI wrapper around the library API."""

import argparse
from brandsafety.api import train_artifact_from_path

def _csv_list(s):
    return [x.strip() for x in (s or '').split(',') if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", required=True, choices=["binary","multiclass"])
    ap.add_argument("--out", required=True)

    ap.add_argument("--label-col", default="label")
    ap.add_argument("--bucket-col", default="sample_bucket")
    ap.add_argument("--embedding-prefix", default="emb_")
    ap.add_argument("--embedding-col", default=None)

    ap.add_argument("--positive", default=None)
    ap.add_argument("--cal-method", default="isotonic", choices=["isotonic","sigmoid"])
    ap.add_argument("--target-precision", type=float, default=0.80)
    ap.add_argument("--per-class-target-precision", type=float, default=None)

    ap.add_argument("--anchor-split", default="stratified", choices=["stratified","group_target_root"])
    ap.add_argument("--group-col", default="target_root")
    ap.add_argument("--anchor-cal-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--category", default="category")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    artifact = train_artifact_from_path(
        args.data,
        mode = args.mode,
        label_col = args.label_col,
        bucket_col = args.bucket_col,
        embedding_prefix = args.embedding_prefix,
        embedding_col = args.embedding_col,
        positive_classes = _csv_list(args.positive) if args.mode == "binary" else None,
        cal_method = args.cal_method,
        target_precision = args.target_precision,
        per_class_target_precision = args.per_class_target_precision,
        anchor_split = args.anchor_split,
        group_col = args.group_col,
        anchor_cal_frac = args.anchor_cal_frac,
        seed = args.seed,
        category = args.category,
        verbose = args.verbose
    )
    artifact.save(args.out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
