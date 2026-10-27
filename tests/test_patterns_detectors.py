import numpy as np
import pandas as pd

from ibm_detectors import DETECTORS, evaluate_detectors, pass_through_closeness, run_detectors, tune_threshold
from ibm_features import past_window_sum
from ibm_patterns import UNATTRIBUTED, label_rows, parse_patterns

PATTERN_TEXT = """BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out
2022/09/01 00:06,021,A1,012,B1,100.00,Euro,100.00,Euro,ACH,1
2022/09/01 04:33,021,A1,020,B2,200.50,Euro,200.50,Euro,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops
2022/09/02 10:00,001,C1,002,C2,50.00,Yuan,50.00,Yuan,Wire,1
END LAUNDERING ATTEMPT - CYCLE
"""


def test_parse_patterns_reads_attempts_and_schemes(tmp_path):
    path = tmp_path / "patterns.txt"
    path.write_text(PATTERN_TEXT)
    p = parse_patterns(path)
    assert len(p) == 3 and p["attempt"].tolist() == [0, 0, 1]
    assert p["scheme"].tolist() == ["FAN-OUT", "FAN-OUT", "CYCLE"]
    assert p["Amount Paid"].tolist() == [100.0, 200.5, 50.0]


def test_label_rows_attaches_schemes_and_flags_unlisted_laundering(tmp_path):
    path = tmp_path / "patterns.txt"
    path.write_text(PATTERN_TEXT)
    raw = pd.DataFrame({
        "Timestamp": ["2022/09/01 00:06", "2022/09/01 04:33", "2022/09/02 10:00", "2022/09/03 11:11", "2022/09/01 00:06"],
        "Account": ["A1", "A1", "C1", "Z9", "Q1"], "Account.1": ["B1", "B2", "C2", "Y9", "R1"],
        "Amount Paid": [100.0, 200.5, 50.0, 5.0, 100.0], "Payment Format": ["ACH", "ACH", "Wire", "ACH", "ACH"],
        "Is Laundering": [1, 1, 1, 1, 0]})
    scheme, attempt = label_rows(raw, parse_patterns(path))
    assert scheme.tolist() == ["FAN-OUT", "FAN-OUT", "CYCLE", UNATTRIBUTED, ""]
    assert attempt.tolist() == [0, 0, 1, -1, -1]


def test_past_window_sum_only_counts_the_window_before_each_row():
    group, t, amount = np.array([0, 0, 0, 0]), np.array([0, 50, 100, 150]), np.array([1.0, 2.0, 4.0, 8.0])
    got = past_window_sum(group, t, amount, group, t, window=60)
    assert got.tolist() == [0.0, 1.0, 2.0, 4.0]          # [t-60, t): row at t=100 sees only the t=50 row


def test_pass_through_closeness_rewards_inflow_that_matches_the_outflow():
    # account 1 receives 90 at t=0, then sends 100 at t=3600: closeness 0.9; no inflow, or 5x too much inflow, scores 0
    src = np.array([0, 1, 5, 7, 8]); dst = np.array([1, 2, 6, 9, 9]); t = np.array([0, 3600, 3600, 0, 3600])
    paid = np.array([90.0, 100.0, 10.0, 500.0, 100.0])
    assert np.allclose(pass_through_closeness(src, dst, t, paid), [0.0, 0.9, 0.0, 0.0, 0.0])


def test_tuned_threshold_uses_fit_rows_only_and_separates_the_classes():
    score = np.array([0, 0, 1, 1, 6, 7, 8, 0, 9, 9], dtype=float)
    target = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    fit = np.arange(10) < 7                                              # the last rows must not influence the choice
    thr = tune_threshold(score, target, fit)
    assert 1 < thr <= 6


def test_run_and_evaluate_detectors_report_precision_and_recall_per_scheme():
    n = 200
    scheme = np.array([''] * n, dtype=object)
    scheme[:20] = 'FAN-OUT'
    y = np.zeros(n, dtype=int)
    y[:20] = 1
    scores = {name: np.zeros(n) for name in DETECTORS}
    scores['fan_out'][:20] = 10.0                                         # a perfect fan-out signal
    train = np.arange(n) < 150
    flags = run_detectors(scores, scheme, y, train)
    report = evaluate_detectors(flags, scheme, y, ~train)
    assert report['fan_out']['flagged'] == 0 and report['fan_out']['own_scheme_rows'] == 0   # all positives are in train
    report_all = evaluate_detectors(flags, scheme, y, np.ones(n, dtype=bool))
    assert report_all['fan_out']['precision_any_laundering'] == 1.0 and report_all['fan_out']['recall_own_schemes'] == 1.0
