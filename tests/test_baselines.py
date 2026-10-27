import numpy as np

from baselines import account_aggregate_features, evaluate, pair_count_feature, tune_threshold


def test_aggregates_use_train_rows_only_and_drop_self_for_train_rows():
    src = np.array([0, 0, 1])
    dst = np.array([1, 2, 0])
    amount = np.array([10.0, 20.0, 5.0])
    train = np.array([True, True, False])
    feats = account_aggregate_features(src, dst, amount, train, n_accounts=3)
    # sender-out count / sum for row 0: account 0 sent 2 training rows (30), minus itself (10) -> 1 row, 20
    assert np.isclose(feats[0, 0], np.log1p(1)) and np.isclose(feats[0, 1], np.log1p(20))
    # row 2 is not a training row and its sender (account 1) sent nothing in training
    assert feats[2, 0] == 0 and feats[2, 1] == 0


def test_aggregates_ignore_evaluation_rows():
    src = np.array([0, 0])
    dst = np.array([1, 1])
    amount = np.array([10.0, 1e9])
    train = np.array([True, False])
    a = account_aggregate_features(src, dst, amount, train, n_accounts=2)
    amount2 = amount.copy()
    amount2[1] = 1.0
    b = account_aggregate_features(src, dst, amount2, train, n_accounts=2)
    assert np.allclose(a[0], b[0])           # a held-out amount cannot change a training row's features


def test_pair_count_is_leave_one_out_for_train_rows():
    src = np.array([0, 0, 0])
    dst = np.array([1, 1, 1])
    train = np.array([True, True, False])
    counts = pair_count_feature(src, dst, train, n_accounts=2)[:, 0]
    assert np.allclose(counts, [np.log1p(1), np.log1p(1), np.log1p(2)])


def test_tuned_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.4])
    thr = tune_threshold(y, p)
    res = evaluate(y, p, thr)
    assert res['precision'] == 1.0 and res['recall'] == 1.0 and res['f1'] == 1.0
    assert res['precision_at_k'] == 1.0 and res['n_pos'] == 2
