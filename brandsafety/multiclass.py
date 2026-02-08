from __future__ import annotations

import json
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

def cv_select_C_by_accuracy(X, y, Cs, seed = 42, n_splits = 5):
    X = X if isinstance(X, np.ndarray) else np.asarray(X)
    y = np.asarray(y)

    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = seed)

    cv_scores = {}
    for C in Cs:
        fold_scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            clf = make_pipeline(LinearSVC(C = C, random_state = seed, multi_class = "ovr"))
            clf.fit(X_tr, y_tr)
            fold_scores.append(accuracy_score(y_va, clf.predict(X_va)))

        cv_scores[float(C)] = float(np.mean(fold_scores))

    best_C = max(cv_scores, key = cv_scores.get)
    return float(best_C), cv_scores


def train_multiclass_ovr(
    X_train, y_train_str,
    X_cal, y_cal_str,
    X_test, y_test_str,
    *,
    classes = None,
    cal_method = "isotonic",
    per_class_target_precision = None,   # kept for API compatibility; unused in legacy multiclass
    prefer = "max_recall",               # kept for API compatibility; unused in legacy multiclass
    seed = 42,
    verbose = False
):
    """
    LEGACY multiclass trainer (single estimator) for compatibility with deployed pipelines.

    - Trains LinearSVC (multi_class='ovr') on multiclass labels (strings or ints)
    - Calibrates with CalibratedClassifierCV(method=cal_method, cv='prefit')
    - Returns (model, metrics_dict)

    Note: The function name is preserved (`train_multiclass_ovr`) to keep repo API stable,
    but the returned artifact now contains a single calibrated estimator (not a dict of per-class models).
    """
    y_train = np.asarray(y_train_str)
    y_cal   = np.asarray(y_cal_str)
    y_test  = np.asarray(y_test_str)

    if classes is None:
        classes = sorted(list(set(y_train.tolist()) | set(y_cal.tolist()) | set(y_test.tolist())))

    C_GRID = (0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 1.0, 3.0, 10.0)
    best_C, cv_scores = cv_select_C_by_accuracy(X_train, y_train, Cs = C_GRID, seed = seed, n_splits = 5)

    Xtr = X_train if isinstance(X_train, np.ndarray) else np.asarray(X_train)
    Xca = X_cal if isinstance(X_cal, np.ndarray) else np.asarray(X_cal)
    Xte = X_test if isinstance(X_test, np.ndarray) else np.asarray(X_test)

    base_pipe = make_pipeline(
        LinearSVC(C = best_C, random_state = seed, multi_class = "ovr")
    )
    base_pipe.fit(Xtr, y_train)

    calibrated = CalibratedClassifierCV(
        base_pipe,
        method = cal_method,
        cv = "prefit"
    )
    calibrated.fit(Xca, y_cal)

    # test eval
    probs = calibrated.predict_proba(Xte)
    y_pred = calibrated.predict(Xte)

    metrics = {
        "best_C": float(best_C),
        "cv_scores_acc": {str(k): float(v) for k, v in cv_scores.items()},
        "cal_method": str(cal_method),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision_macro": float(precision_score(y_test, y_pred, average = "macro", zero_division = 0)),
        "test_recall_macro": float(recall_score(y_test, y_pred, average = "macro", zero_division = 0)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average = "macro", zero_division = 0)),
        "test_log_loss": float(log_loss(y_test, probs)),
    }

    if verbose:
        cm = confusion_matrix(y_test, y_pred, labels = calibrated.classes_)
        metrics["confusion_matrix"] = cm.tolist()
        print("Selected C:", best_C)
        print("Metrics:", json.dumps(metrics, indent = 2))

    return calibrated, metrics
