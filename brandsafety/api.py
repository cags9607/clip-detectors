from __future__ import annotations
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from .data import load_table, extract_embeddings
from .split import split_by_bucket
from .binary import train_binary
from .multiclass import train_multiclass_ovr
from .artifact import Artifact, load_artifact as _load_artifact
from .artifacts_io import save as _save

def load_artifact(path: str) -> Artifact:
    return _load_artifact(path)

def train_artifact(
    df: pd.DataFrame,
    *,
    mode: str,
    label_col: str = "label",
    bucket_col: str = "sample_bucket",
    anchor_value: str = "anchor_unbiased",
    train_values = ("mined_keywords", "diversity_long_tail"),
    embedding_prefix: str = "emb_",
    embedding_col: Optional[str] = None,
    # split
    anchor_split: str = "stratified",
    group_col: str = "target_root",
    anchor_cal_frac: float = 0.5,
    seed: int = 42,
    # calibration & thresholds
    cal_method: str = "isotonic",
    target_precision: float = 0.80,
    prefer: str = "max_recall",
    # binary mode only
    positive_classes: Optional[List[str]] = None,
    # multiclass mode only (optional)
    per_class_target_precision: Optional[float] = None,
    # metadata
    category: str = "category",
    verbose: bool = True,
) -> Artifact:
    if mode not in ("binary", "multiclass"):
        raise ValueError("mode must be 'binary' or 'multiclass'")

    df_train, df_cal, df_test = split_by_bucket(
        df,
        label_col = label_col,
        bucket_col = bucket_col,
        anchor_value = anchor_value,
        train_values = tuple(train_values),
        anchor_cal_frac = anchor_cal_frac,
        seed = seed,
        anchor_split = anchor_split,
        group_col = group_col
    )

    X_train = extract_embeddings(df_train, embedding_prefix = embedding_prefix, embedding_col = embedding_col)

    if len(df_cal) > 0:
        X_cal = extract_embeddings(df_cal, embedding_prefix = embedding_prefix, embedding_col = embedding_col)
    else:
        X_cal = np.empty((0, X_train.shape[1]), dtype = np.float32)

    X_test = extract_embeddings(df_test, embedding_prefix = embedding_prefix, embedding_col = embedding_col)

    y_train = df_train[label_col].astype(str).values
    y_cal   = df_cal[label_col].astype(str).values
    y_test  = df_test[label_col].astype(str).values

    obj: Dict[str, Any] = {
        "category": category,
        "task": None,
        "meta": {
            "mode": mode,
            "label_col": label_col,
            "bucket_col": bucket_col,
            "anchor_value": anchor_value,
            "train_values": list(train_values),
            "anchor_split": anchor_split,
            "group_col": group_col,
            "anchor_cal_frac": float(anchor_cal_frac),
            "seed": int(seed),
            "cal_method": cal_method,
            "target_precision": float(target_precision),
            "prefer": prefer,
            "feature_spec": {"embedding_prefix": embedding_prefix, "embedding_col": embedding_col},
            "sizes": {"train": int(len(df_train)), "cal": int(len(df_cal)), "test": int(len(df_test))},
        }
    }

    if mode == "binary":
        if not positive_classes:
            raise ValueError("positive_classes is required for binary mode")
        pos = set([str(x) for x in positive_classes])

        ytr = np.isin(y_train, list(pos)).astype(int)
        yca = np.isin(y_cal,   list(pos)).astype(int)
        yte = np.isin(y_test,  list(pos)).astype(int)

        model, thr, metrics = train_binary(
            X_train, ytr,
            X_cal, yca,
            X_test, yte,
            cal_method = cal_method,
            target_precision = target_precision,
            prefer = prefer,
            seed = seed,
            verbose = verbose
        )

        obj.update({
            "task": "binary",
            "classes": ["neg", "pos"],
            "positive_classes": sorted(list(pos)),
            "model": model,
            "threshold": float(thr),
            "metrics": metrics
        })

    else:
        classes = sorted(list(set(y_train) | set(y_cal) | set(y_test)))
        model, metrics = train_multiclass_ovr(
            X_train, y_train,
            X_cal, y_cal,
            X_test, y_test,
            classes = classes,
            cal_method = cal_method,
            per_class_target_precision = per_class_target_precision,
            prefer = prefer,
            seed = seed,
            verbose = verbose
        )

        obj.update({
            "task": "multiclass",
            "classes": list(getattr(model, "classes_", classes)),
            "model": model,
            "metrics": metrics
        })

    return Artifact(obj)

def train_artifact_from_path(path: str, **kwargs) -> Artifact:
    df = load_table(path)
    return train_artifact(df, **kwargs)
