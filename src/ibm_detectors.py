"""
Rule-based pattern detectors for IBM transactions (smurfing / layering / ring shapes), plus how well they do.

Each detector turns one past-only signal into a score and flags a transaction when the score reaches a threshold.
Thresholds are tuned on the TRAINING period only (best F1 against the benchmark scheme labels the detector targets) and
then applied unchanged to validation and test. Detectors see only information from strictly earlier timestamps.
"""
import numpy as np

from ibm_features import GRAPH_FEATURE_NAMES, FEATURE_NAMES, past_window_sum

ALL_NAMES = FEATURE_NAMES + GRAPH_FEATURE_NAMES
DAY = 86400

# label shown in the UI, benchmark schemes each detector is meant to catch
DETECTORS = {
    'fan_out': dict(label='Fan-out (smurfing source)', schemes=['FAN-OUT']),
    'fan_in': dict(label='Fan-in (smurfing target)', schemes=['FAN-IN']),
    'cycle': dict(label='Cycle (ring)', schemes=['CYCLE']),
    'pass_through': dict(label='Pass-through (layering)', schemes=['STACK', 'SCATTER-GATHER', 'GATHER-SCATTER']),
}


def _col(X, name):
    return np.expm1(X[:, ALL_NAMES.index(name)].astype(float))     # features are stored as log1p


def pass_through_closeness(src, dst, t, paid):
    """
    How closely what the sender received in the previous 24 hours matches what it sends now: 1 when the amounts are
    equal, 0 when nothing came in or the inflow is at least double (or at most zero) the outflow. High means the account
    is passing money straight through.
    """
    received = past_window_sum(dst, t, paid, src, t, DAY)
    return np.maximum(0.0, 1.0 - np.abs(received / np.maximum(paid, 1e-9) - 1.0))


def detector_scores(X, src, dst, t, paid):
    """Score per detector for every row. X is the 43-column past-only feature matrix."""
    return {
        'fan_out': _col(X, 'sender_out_new_receivers_1d'),
        'fan_in': _col(X, 'receiver_in_new_senders_1d'),
        'cycle': _col(X, 'three_cycle_witnesses'),
        'pass_through': pass_through_closeness(src, dst, t, paid),
    }


def tune_threshold(score, target, fit_mask):
    """
    Threshold with the best F0.5 against `target` on the fit rows (precision counts double, so a badge is not attached to
    a huge share of ordinary transactions). Candidates are the distinct high score values.
    """
    s, y = score[fit_mask], target[fit_mask].astype(bool)
    if not y.any():
        return float('inf')
    candidates = np.unique(np.quantile(s[s > 0], np.linspace(0.5, 0.9999, 60))) if (s > 0).any() else np.array([1.0])
    best, best_f = float(candidates[-1]), -1.0
    positives = int(y.sum())
    for thr in candidates:
        flag = s >= thr
        tp = int((flag & y).sum())
        fp = int(flag.sum()) - tp
        f = 1.25 * tp / max(1.25 * tp + 0.25 * (positives - tp) + fp, 1e-9)
        if f > best_f:
            best, best_f = float(thr), f
    return best


def run_detectors(scores, scheme, y, train_mask):
    """Tune each detector on training rows; returns {name: (threshold, flags)}."""
    out = {}
    for name, info in DETECTORS.items():
        target = np.isin(scheme, info['schemes'])
        if not target[train_mask].any():                     # no scheme labels: fall back to the laundering flag
            target = y == 1
        thr = tune_threshold(scores[name], target, train_mask)
        out[name] = (thr, scores[name] >= thr)
    return out


def evaluate_detectors(flags, scheme, y, mask):
    """Precision against the laundering flag and recall against each detector's own schemes, on the rows in `mask`."""
    report = {}
    for name, (thr, flag) in flags.items():
        f = flag[mask]
        yy = y[mask] == 1
        own = np.isin(scheme[mask], DETECTORS[name]['schemes'])
        report[name] = dict(
            label=DETECTORS[name]['label'], threshold=float(thr), flagged=int(f.sum()),
            precision_any_laundering=float((f & yy).sum() / max(f.sum(), 1)),
            own_scheme_rows=int(own.sum()), recall_own_schemes=float((f & own).sum() / max(own.sum(), 1)),
            recall_all_laundering=float((f & yy).sum() / max(yy.sum(), 1)),
        )
    return report
