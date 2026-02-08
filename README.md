# deepsee-legacy-svm

This repo exports **legacy-compatible** `.pkl` files for your current pipeline.

- Binary and multiclass LinearSVC
- Isotonic calibration (`CalibratedClassifierCV(cv="prefit")`)
- **The saved .pkl is the raw sklearn estimator** (no dicts, no wrappers)

## Install

```bash
pip install -e .
```

## Binary

```python
from deepsee_legacy_svm import train_binary_legacy
model, metrics = train_binary_legacy(
    X_train, y_train,
    X_cal, y_cal,
    X_test, y_test,
    out_pkl_path="svm_calibrated_binary.pkl",
)
```

## Multiclass

Encode labels externally (same as your notebook):

```python
from sklearn.preprocessing import LabelEncoder
from deepsee_legacy_svm import train_multiclass_legacy

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_raw)
y_cal_enc   = le.transform(y_cal_raw)
y_test_enc  = le.transform(y_test_raw)

model, metrics = train_multiclass_legacy(
    X_train, y_train_enc,
    X_cal, y_cal_enc,
    X_test, y_test_enc,
    out_pkl_path="svm_calibrated_multiclass.pkl",
)
```
