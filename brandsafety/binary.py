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
    y_cal   = np.asarray(y_cal).astype(int)
    y_test  = np.asarray(y_test).astype(int)

    C_GRID = (0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 1, 3, 10)
    best_C, cv_scores = cv_select_C_by_ap(X_train, y_train, Cs = C_GRID, seed = seed)

    base_pipe = make_pipeline(LinearSVC(C = best_C, random_state = seed))
    base_pipe.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(base_pipe, method = cal_method, cv = "prefit")
    calibrated.fit(X_cal, y_cal)

    p_cal  = calibrated.predict_proba(X_cal)[:, 1]
    p_test = calibrated.predict_proba(X_test)[:, 1]

    thr, P_cal, R_cal, F1_cal, how = threshold_for_precision(
        y_cal, p_cal, target_precision = target_precision, prefer = prefer
    )

    yhat_test = (p_test >= 0.5).astype(int) # use thr instead of 0.5 for target precision
    tn, fp, fn, tp = confusion_matrix(y_test, yhat_test).ravel()
    fpr = fp / (fp + tn + 1e-12)

    metrics = dict(
        best_C = float(best_C),
        cv_scores_ap = cv_scores,
        cal_method = cal_method,
        target_precision = float(target_precision),
        thr = float(thr),
        how = how,
        cal_precision = float(P_cal),
        cal_recall = float(R_cal),
        cal_f1 = float(F1_cal),
        test_precision = float(precision_score(y_test, yhat_test, zero_division = 0)),
        test_recall = float(recall_score(y_test, yhat_test, zero_division = 0)),
        test_f1 = float(f1_score(y_test, yhat_test, zero_division = 0)),
        test_fpr = float(fpr),
        test_roc_auc = float(roc_auc_score(y_test, p_test)),
        test_pr_auc = float(average_precision_score(y_test, p_test)),
        test_ece = float(ece_equal_width(p_test, y_test, n_bins = 15)),
        n_train = int(len(y_train)),
        n_cal = int(len(y_cal)),
        n_test = int(len(y_test)),
    )

    if verbose:
        print(json.dumps(metrics, indent = 2))

    return calibrated, float(thr), metrics
