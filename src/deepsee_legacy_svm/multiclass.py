import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

def train_multiclass_legacy(
    X_train, y_train,
    X_cal, y_cal,
    X_test, y_test,
    *,
    C=0.5,
    out_pkl_path="svm_calibrated_multiclass.pkl",
):
    X_tr = X_train.values if hasattr(X_train, "values") else X_train
    X_ca = X_cal.values if hasattr(X_cal, "values") else X_cal
    X_te = X_test.values if hasattr(X_test, "values") else X_test

    base = make_pipeline(
        LinearSVC(C=C, random_state=42, multi_class="ovr")
    )
    base.fit(X_tr, y_train)

    clf = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    clf.fit(X_ca, y_cal)

    probs = clf.predict_proba(X_te)
    y_pred = clf.predict(X_te)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, probs)),
    }

    with open(out_pkl_path, "wb") as f:
        pickle.dump(clf, f)

    return clf, metrics
