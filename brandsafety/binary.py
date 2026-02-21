import json
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

from .metrics import ece_equal_width, threshold_for_precision

def cv_select_C_by_ap(X, y, Cs, seed = 42, n_splits = 5):
    X = X if isinstance(X, np.ndarray) else np.asarray(X)
    y = np.asarray(y).astype(int)

    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = seed)

    cv_scores = {}
    for C in Cs:
        fold_scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            clf = make_pipeline(LinearSVC(C = C, random_state = seed))
            clf.fit(X_tr, y_tr)

            scores = clf.decision_function(X_va)
            fold_scores.append(average_precision_score(y_va, scores))

        cv_scores[float(C)] = float(np.mean(fold_scores))

    best_C = max(cv_scores, key = cv_scores.get)
    return best_C, cv_scores

def train_binary(
    X_train, y_train,
    X_cal, y_cal,
    X_test, y_test,
    *,
    cal_method = "isotonic",
    target_precision = 0.80,
    prefer = "max_recall",
    seed = 42,
    verbose = False
):
    y_train = np.asarray(y_train).astype(int)
    y_test  = np.asarray(y_test).astype(int)

    has_cal = (X_cal is not None) and (y_cal is not None) and (len(y_cal) > 0)

    if has_cal:
        y_cal = np.asarray(y_cal).astype(int)
    else:
        y_cal = np.asarray([]).astype(int)

    C_GRID = (0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 1, 3, 10)
    best_C, cv_scores = cv_select_C_by_ap(X_train, y_train, Cs = C_GRID, seed = seed)

    base_pipe = make_pipeline(LinearSVC(C = best_C, random_state = seed))

    if has_cal:
        base_pipe.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(base_pipe, method = cal_method, cv = "prefit")
        calibrated.fit(X_cal, y_cal)
        p_cal = calibrated.predict_proba(X_cal)[:, 1]
    else:
        calibrated = CalibratedClassifierCV(base_pipe, method = cal_method, cv = 5)
        calibrated.fit(X_train, y_train)
        p_cal = None

    p_test = calibrated.predict_proba(X_test)[:, 1]

    if has_cal:
        thr, P_cal, R_cal, F1_cal, how = threshold_for_precision(
            y_cal, p_cal, target_precision = target_precision, prefer = prefer
        )
        thr = 0.5 # Default to 0.5 just use calibration to produce scores 
    else:
        thr = 0.5
        P_cal, R_cal, F1_cal = float("nan"), float("nan"), float("nan")
        how = "no_cal_default_thr_0.5"

    yhat_test = (p_test >= float(thr)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat_test, labels = [0, 1]).ravel()
    fpr = fp / (fp + tn + 1e-12)

    def _safe(fn, default = float("nan")):
        try:
            return float(fn())
        except Exception:
            return default

    metrics = dict(
        best_C = float(best_C),
        cv_scores_ap = cv_scores,
        cal_method = cal_method,
        target_precision = float(target_precision),
        thr = float(thr),
        how = how,
        cal_precision = float(P_cal) if P_cal == P_cal else float("nan"),
        cal_recall = float(R_cal) if R_cal == R_cal else float("nan"),
        cal_f1 = float(F1_cal) if F1_cal == F1_cal else float("nan"),
        test_precision = float(precision_score(y_test, yhat_test, zero_division = 0)),
        test_recall = float(recall_score(y_test, yhat_test, zero_division = 0)),
        test_f1 = float(f1_score(y_test, yhat_test, zero_division = 0)),
        test_fpr = float(fpr),
        test_roc_auc = _safe(lambda: roc_auc_score(y_test, p_test)),
        test_pr_auc = _safe(lambda: average_precision_score(y_test, p_test)),
        test_ece = float(ece_equal_width(p_test, y_test, n_bins = 15)),
        n_train = int(len(y_train)),
        n_cal = int(len(y_cal)),
        n_test = int(len(y_test)),
    )

    if verbose:
        print(json.dumps(metrics, indent = 2))

    return calibrated, float(thr), metrics
