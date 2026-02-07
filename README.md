# brandsafety-svm

Minimal training + inference repo for brand-safety image classifiers using **precomputed embeddings** and an **SVM + calibration** pipeline.

Supports:
- **Binary** training (unsafe vs safe), with a stored threshold for a target precision.
- **Multiclass (mutually exclusive)** training via **One-vs-Rest (OvR)** calibrated models.

Assumes you upload your own labeled dataset containing:
- `sample_bucket` (e.g., `anchor_unbiased`, `mined_keywords`, `diversity_long_tail`)
- `match_reason` (string; diagnostic only)
- `label` (string class, mutually exclusive)
- embeddings:
  - Option A: wide columns `emb_0..emb_d` (recommended)
  - Option B: single column `embedding` with list/np array

## Install

```bash
pip install -r requirements.txt
```

## Training

### Binary
Example: treat `explicit` as positive.

```bash
python train.py   --data labeled_embeddings.parquet   --mode binary   --label-col label   --bucket-col sample_bucket   --embedding-prefix emb_   --positive explicit   --target-precision 0.80   --cal-method isotonic   --anchor-split stratified   --anchor-cal-frac 0.5   --out nudity_binary.pkl
```

### Multiclass (mutually exclusive)
OvR calibrated models.

```bash
python train.py   --data labeled_embeddings.parquet   --mode multiclass   --label-col label   --bucket-col sample_bucket   --embedding-prefix emb_   --cal-method isotonic   --anchor-split stratified   --anchor-cal-frac 0.5   --out nudity_multiclass.pkl
```

Optional: compute per-class thresholds for a target precision (on anchor-cal):

```bash
python train.py   --data labeled_embeddings.parquet   --mode multiclass   --label-col label   --bucket-col sample_bucket   --embedding-prefix emb_   --cal-method isotonic   --target-precision 0.90   --out nudity_multiclass.pkl
```

## Inference

```bash
python infer.py   --model nudity_multiclass.pkl   --data new_embeddings.parquet   --embedding-prefix emb_   --out scored.parquet
```

Outputs:
- Binary: `p_hat`, and `yhat` if threshold exists in the artifact.
- Multiclass: `prob_<class>` for each class and `pred_label` (argmax).

## Methodology

- Train pool: `mined_keywords + diversity_long_tail`
- Calibration set: `anchor_unbiased` split into cal/test
- Threshold selection: on **anchor-cal**
- Metrics: reported on **anchor-test**

