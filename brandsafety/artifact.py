from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from .data import extract_embeddings
from .artifacts_io import save as _save, load as _load

@dataclass
class Artifact:
    obj: Dict[str, Any]

    def save(self, path: str) -> None:
        """Save a LEGACY-compatible .pkl: the raw sklearn estimator only."""
        task = self.task
        if task == "binary":
            _save(self.obj["model"], path)
            return
        if task == "multiclass":
            _save(self.obj["model"], path)
            return
        # fallback: old behavior
        _save(self.obj, path)

    @property
    def task(self) -> str:
        return str(self.obj.get("task"))

    def predict_proba(self, X: np.ndarray) -> Dict[str, Any]:
        task = self.task
        if task == "binary":
            model = self.obj["model"]
            p_hat = model.predict_proba(X)[:, 1]
            thr = self.obj.get("threshold", None)
            return {"p_hat": p_hat, "threshold": thr}
        if task == "multiclass":
            model = self.obj["model"]
            probs = model.predict_proba(X)
            classes = list(getattr(model, "classes_", []))
            return {"probs": probs, "classes": classes}
        raise ValueError(f"Unknown task: {task}")

def load_artifact(path: str) -> Artifact:
    return Artifact(obj = _load(path))

def predict_embeddings(artifact: Artifact, emb: np.ndarray) -> Dict[str, Any]:
    return artifact.predict_proba(emb)

def predict_df(artifact: Artifact, df: pd.DataFrame, *, embedding_prefix: str = "emb_", embedding_col: str | None = None) -> Dict[str, Any]:
    X = extract_embeddings(df, embedding_prefix = embedding_prefix, embedding_col = embedding_col)
    return artifact.predict_proba(X)
