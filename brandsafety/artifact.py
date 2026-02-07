from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from .data import extract_embeddings
from .artifacts_io import save as _save, load as _load

@dataclass
class Artifact:
    obj: Dict[str, Any]

    def save(self, path: str) -> None:
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
        if task == "multiclass_ovr":
            classes = self.obj["classes"]
            models = self.obj["models"]
            probs = {}
            for c in classes:
                if c in models:
                    probs[c] = models[c].predict_proba(X)[:, 1]
                else:
                    probs[c] = np.full(shape = (X.shape[0],), fill_value = -1.0, dtype = float)
            return {"probs": probs, "classes": classes}
        raise ValueError(f"Unknown task: {task}")

    def predict_df(self, df: pd.DataFrame, *, embedding_prefix: str = "emb_", embedding_col: Optional[str] = None) -> pd.DataFrame:
        X = extract_embeddings(df, embedding_prefix = embedding_prefix, embedding_col = embedding_col)
        out_df = df.copy()

        task = self.task
        if task == "binary":
            res = self.predict_proba(X)
            p_hat = res["p_hat"]
            out_df["p_hat"] = p_hat
            thr = res.get("threshold", None)
            if thr is not None:
                out_df["yhat"] = (p_hat >= float(thr)).astype(int)
            return out_df

        if task == "multiclass_ovr":
            res = self.predict_proba(X)
            classes = res["classes"]
            probs = res["probs"]
            for c in classes:
                out_df[f"prob_{c}"] = probs[c]
            M = np.vstack([probs[c] for c in classes]).T
            pred_idx = np.argmax(M, axis = 1)
            out_df["pred_label"] = [classes[i] for i in pred_idx]
            return out_df

        raise ValueError(f"Unknown task: {task}")

def load_artifact(path: str) -> Artifact:
    return Artifact(_load(path))
