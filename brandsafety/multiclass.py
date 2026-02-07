import json
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

from .binary import train_binary

def train_multiclass_ovr(
    X_train, y_train_str,
    X_cal, y_cal_str,
    X_test, y_test_str,
    *,
    classes = None,
    cal_method = "isotonic",
    per_class_target_precision = None,
    prefer = "max_recall",
    seed = 42,
    verbose = False
):
    y_train_str = np.asarray(y_train_str).astype(str)
    y_cal_str   = np.asarray(y_cal_str).astype(str)
    y_test_str  = np.asarray(y_test_str).astype(str)

    if classes is None:
        classes = sorted(list(set(y_train_str) | set(y_cal_str) | set(y_test_str)))

    models = {}
    thresholds = {} if per_class_target_precision is not None else None
    per_class_metrics = {}

    for c in classes:
        ytr = (y_train_str == c).astype(int)
        yca = (y_cal_str == c).astype(int)
        yte = (y_test_str == c).astype(int)

        if ytr.sum() == 0 or yca.sum() == 0:
            continue

        tp = float(per_class_target_precision) if per_class_target_precision is not None else 0.80

        model_c, thr_c, met_c = train_binary(
            X_train, ytr,
            X_cal, yca,
            X_test, yte,
            cal_method = cal_method,
            target_precision = tp,
            prefer = prefer,
            seed = seed,
            verbose = False
        )

        models[c] = model_c
        per_class_metrics[c] = met_c
        if thresholds is not None:
            thresholds[c] = float(thr_c)

    # multiclass preds via argmax of per-class probs; missing classes -> -inf
    proba_test = {}
    for c in classes:
        if c in models:
            proba_test[c] = models[c].predict_proba(X_test)[:, 1]
        else:
            proba_test[c] = np.full(shape = (X_test.shape[0],), fill_value = -1.0, dtype = float)

    M = np.vstack([proba_test[c] for c in classes]).T
    pred_idx = np.argmax(M, axis = 1)
    y_pred = np.asarray([classes[i] for i in pred_idx], dtype = object)

    acc = float(accuracy_score(y_test_str, y_pred))
    cm = confusion_matrix(y_test_str, y_pred, labels = classes).tolist()

    # diagnostic: softmax-normalized log loss (OvR probs don't sum to 1)
    M2 = M.copy()
    M2[M2 < 0] = -20.0
    ex = np.exp(M2 - M2.max(axis = 1, keepdims = True))
    Pn = ex / (ex.sum(axis = 1, keepdims = True) + 1e-12)

    try:
        ll = float(log_loss(y_test_str, Pn, labels = classes))
    except Exception:
        ll = None

    metrics = dict(
        task = "multiclass_ovr",
        classes = classes,
        n_models = int(len(models)),
        test_accuracy = acc,
        test_log_loss_softmax_diag = ll,
        confusion_matrix = cm,
        per_class = per_class_metrics
    )

    if verbose:
        print(json.dumps(metrics, indent = 2))

    return models, thresholds, metrics
