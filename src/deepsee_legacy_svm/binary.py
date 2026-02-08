import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score

def train_binary_legacy(
    X_train, y_train,
    X_cal, y_cal,
    X_test, y_test,
    *,
    C=0.5,
    out_pkl_path="svm_calibrated_binary.pkl",
):
    X_tr = X_train.values if hasattr(X_train, "values") else X_train
    X_ca = X_cal.values if hasattr(X_cal, "values") else X_cal
    X_te = X_test.values if hasattr(X_test, "values") else X_test

    base = make_pipeline(LinearSVC(C=C, random_state=42))
    base.fit(X_tr, y_train)

    clf = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    clf.fit(X_ca, y_cal)

    y_pred = clf.predict(X_te)
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    with open(out_pkl_path, "wb") as f:
        pickle.dump(clf, f)

    return clf, metrics
