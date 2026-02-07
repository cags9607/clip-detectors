# brandsafety-svm

A tiny **Python package** (pip-installable) to train and run brand-safety image classifiers using **precomputed embeddings** with an **SVM + probability calibration** pipeline.

Supports:
- **Binary** training (unsafe vs safe), with a stored threshold for a target precision.
- **Multiclass (mutually exclusive)** training via **One-vs-Rest (OvR)** calibrated models.

## Install

From the repo folder:

```bash
pip install -e .
```

If you want Parquet support:

```bash
pip install -e ".[parquet]"
```

## Data contract

Your dataset should include:
- `label` (string class; mutually exclusive)
- `sample_bucket` (e.g., `anchor_unbiased`, `mined_keywords`, `diversity_long_tail`)
- embeddings:
  - Option A (recommended): wide columns `emb_0..emb_d` with a common prefix (default: `emb_`)
  - Option B: a single column `embedding` containing list/np array

Optional columns are kept as metadata and can be preserved in inference outputs:
- `match_reason`, `target_root`, `orig_image_url`, etc.

Input formats:
- **CSV** is always supported
- **Parquet** supported if you install the `parquet` extra

## Library usage

### Train (binary)

```python
import pandas as pd
from brandsafety import train_artifact

df = pd.read_csv("labeled_embeddings.csv")

artifact = train_artifact(
    df,
    mode = "binary",
    label_col = "label",
    bucket_col = "sample_bucket",
    embedding_prefix = "emb_",
    positive_classes = ["explicit"],   # binary mode only
    cal_method = "isotonic",
    target_precision = 0.80,
    anchor_split = "stratified",
    anchor_cal_frac = 0.5,
    seed = 42,
)

artifact.save("nudity_binary.pkl")   # deployment artifact
```

### Train (multiclass OvR)

```python
from brandsafety import train_artifact

artifact = train_artifact(
    df,
    mode = "multiclass",
    label_col = "label",
    bucket_col = "sample_bucket",
    embedding_prefix = "emb_",
    cal_method = "isotonic",
    # optional per-class thresholds (computed on anchor-cal)
    per_class_target_precision = 0.90,
)

artifact.save("nudity_multiclass.pkl")
```

### Inference

```python
import pandas as pd
from brandsafety import load_artifact

artifact = load_artifact("nudity_multiclass.pkl")

df_new = pd.read_csv("new_embeddings.csv")
scored = artifact.predict_df(df_new, embedding_prefix = "emb_")

# scored contains:
# - binary: p_hat (+ yhat if threshold stored)
# - multiclass: prob_<class> columns + pred_label
scored.to_parquet("scored.parquet", index = False)
```

## Notes on methodology

- Train pool: `mined_keywords + diversity_long_tail`
- Calibration: `anchor_unbiased` split into cal/test
- Threshold selection: on **anchor-cal**
- Reporting: metrics on **anchor-test**

