import numpy as np
from sklearn.metrics import precision_recall_curve

def ece_equal_width(p_hat, y_true, n_bins = 15):
    p_hat = np.asarray(p_hat, dtype = float)
    y_true = np.asarray(y_true, dtype = int)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (p_hat >= lo) & (p_hat < hi) if i < n_bins - 1 else (p_hat >= lo) & (p_hat <= hi)
        if not np.any(in_bin):
            continue

        bin_conf = p_hat[in_bin].mean()
        bin_acc  = y_true[in_bin].mean()
        bin_w    = in_bin.mean()

        out += bin_w * abs(bin_acc - bin_conf)

    return float(out)

def threshold_for_precision(y_true, p_hat, target_precision = 0.80, prefer = "max_recall"):
    y_true = np.asarray(y_true).astype(int)
    p_hat  = np.asarray(p_hat).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, p_hat)

    P = precision[1:]
    R = recall[1:]
    thr = thresholds

    ok = (P >= target_precision)
    if not np.any(ok):
        j = int(np.argmax(P))
        thr_best = float(thr[j])
        P_best, R_best = float(P[j]), float(R[j])
        F1_best = float((2 * P_best * R_best) / (P_best + R_best + 1e-12))
        return thr_best, P_best, R_best, F1_best, "no_thr_meets_target_precision"

    idx = np.where(ok)[0]

    if prefer == "max_recall":
        j = idx[np.argmax(R[idx])]
        how = "max_recall_given_precision"
    elif prefer == "min_threshold":
        j = idx[np.argmin(thr[idx])]
        how = "min_threshold_meeting_precision"
    else:
        f1 = (2 * P * R) / (P + R + 1e-12)
        j = idx[np.argmax(f1[idx])]
        how = "max_f1_given_precision"

    thr_best = float(thr[j])
    P_best, R_best = float(P[j]), float(R[j])
    F1_best = float((2 * P_best * R_best) / (P_best + R_best + 1e-12))
    return thr_best, P_best, R_best, F1_best, how
