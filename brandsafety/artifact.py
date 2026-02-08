# brandsafety/artifact.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .data import extract_embeddings
from .artifacts_io import save as _save, load as _load


@dataclass
class Artifact:
    """
    Runtime wrapper around an artifact object.

    Historically, the repo stored a dict-like artifact:
      {
        "task": "binary" | "multiclass",
        "model": <sklearn estimator>,
        ...
      }

    For legacy pipeline compatibility, you may also load a .pkl that contains
    ONLY the raw sklearn estimator. In that case, load_artifact() wraps it into
    the dict-like shape above.
    """
    obj: Dict[str, Any]

    @property
    def task(self) -> str:
        return str(self.obj.get("task"))

    def save(self, path: str) -> None:
        """
        Save a LEGACY-compatible .pkl: the raw sklearn estimator only.
        This keeps compatibility with your deployed pipeline.
        """
        task = self.task
        if task in ("binary", "multiclass"):
            _save(self.obj["model"], path)
            return

        # fallback (shouldn't happen in normal usage)
        _save(self.obj, path)

    def predict_proba(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Standardized prediction output.
        - binary: returns {"p_hat": (n,), "threshold": thr}
        - multiclass: returns {"probs": (n, K), "classes": [...]}
        """
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
            classes = [str(c) for c in classes]
            return {"probs": probs, "classes": classes}

        raise ValueError(f"Unknown task: {task}")

    def predict_embeddings(self, emb: np.ndarray) -> Dict[str, Any]:
        return self.predict_proba(emb)

    def predict_df(
        self,
        df: pd.DataFrame,
        *,
        embedding_prefix: str = "emb_",
        embedding_col: Optional[str] = None
    ) -> Dict[str, Any]:
        X = extract_embeddings(df, embedding_prefix = embedding_prefix, embedding_col = embedding_col)
        return self.predict_proba(X)


def load_artifact(path: str) -> Artifact:
    """
    Loads either:
    1) New/old dict-like artifact (repo-native)
    2) Legacy raw sklearn estimator (CalibratedClassifierCV / Pipeline / etc)

    In case (2), we wrap into a dict-like artifact so the rest of the repo API
    continues to work unchanged.
    """
    obj = _load(path)

    # Legacy .pkl compatibility: raw estimator, not a dict
    if not isinstance(obj, dict):
        model = obj
        classes = list(getattr(model, "classes_", []))
        task = "multiclass" if len(classes) > 2 else "binary"

        wrapped = {
            "task": task,
            "model": model,
        }

        # Add classes for convenience
        if classes:
            wrapped["classes"] = [str(c) for c in classes]

        return Artifact(obj = wrapped)

    # Repo-native artifact dict
    return Artifact(obj = obj)


# -------------------------------------------------------------------
# Backwards-compatible functional helpers (if used elsewhere)
# -------------------------------------------------------------------
def predict_embeddings(artifact: Artifact, emb: np.ndarray) -> Dict[str, Any]:
    return artifact.predict_embeddings(emb)


def predict_df(
    artifact: Artifact,
    df: pd.DataFrame,
    *,
    embedding_prefix: str = "emb_",
    embedding_col: Optional[str] = None
) -> Dict[str, Any]:
    return artifact.predict_df(df, embedding_prefix = embedding_prefix, embedding_col = embedding_col)
