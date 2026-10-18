import numpy as np
import pandas as pd

from ibm_features import FEATURE_NAMES, build_past_only_features, past_stats


def test_past_stats_counts_only_strictly_earlier_rows_of_the_same_group():
    group = np.array([0, 0, 0, 1, 0])
    t = np.array([10, 20, 20, 15, 100])
    amount = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    count, total, since, (w60,) = past_stats(group, t, amount, group, t, windows=(60,))
    assert count.tolist() == [0, 1, 1, 0, 3]            # same-time rows (row 1 vs 2) do not see each other
    assert total.tolist() == [0.0, 1.0, 1.0, 0.0, 7.0]
    assert since[1] == 10 and since[4] == 80
    assert w60.tolist() == [0, 1, 1, 0, 0]              # only rows in the last 60s: row 4's earlier ones are 80s+ old


def test_past_stats_is_unaffected_by_later_rows():
    group = np.array([0, 0, 0])
    t = np.array([10, 20, 30])
    a = past_stats(group, t, np.array([1.0, 2.0, 3.0]), group, t)
    b = past_stats(group, t, np.array([1.0, 2.0, 999.0]), group, t)
    for first, second in zip(a[:3], b[:3]):
        assert np.allclose(first[:2], second[:2])         # rows 0 and 1 cannot see row 2


def _frame():
    rng = np.random.default_rng(0)
    n = 60
    ts = pd.date_range("2022-09-01", periods=n, freq="30min").strftime("%Y/%m/%d %H:%M").to_numpy()
    df = pd.DataFrame({
        "Timestamp": ts,
        "Account": rng.integers(0, 6, n),
        "Account.1": rng.integers(0, 6, n),
        "Amount Paid": rng.uniform(1, 100, n).round(2),
        "Payment Format": rng.choice(["Cheque", "ACH", "Wire"], n),
    })
    return df


def test_features_do_not_depend_on_future_transactions():
    df = _frame()
    src, dst = df["Account"].to_numpy(), df["Account.1"].to_numpy()
    base, names = build_past_only_features(df, src, dst)
    assert names == FEATURE_NAMES and base.shape == (len(df), len(FEATURE_NAMES))

    cut = 40
    altered = df.copy()
    altered.loc[cut:, "Amount Paid"] = altered.loc[cut:, "Amount Paid"] * 50
    altered.loc[cut:, "Account"] = (altered.loc[cut:, "Account"] + 1) % 6
    other, _ = build_past_only_features(altered, altered["Account"].to_numpy(), dst)
    assert np.allclose(base[:cut], other[:cut])           # nothing at or after row 40 touches earlier rows


def test_features_keep_input_row_order_when_rows_are_shuffled():
    df = _frame()
    src, dst = df["Account"].to_numpy(), df["Account.1"].to_numpy()
    base, _ = build_past_only_features(df, src, dst)
    perm = np.random.default_rng(3).permutation(len(df))
    shuffled, _ = build_past_only_features(df.iloc[perm].reset_index(drop=True), src[perm], dst[perm])
    assert np.allclose(shuffled, base[perm])


# ----------------------------------------------------------------- graph-structure features
from ibm_features import GRAPH_FEATURE_NAMES, build_past_only_graph_features, three_cycle_witnesses  # noqa: E402


def _tx(rows):
    """rows: (minute, sender, receiver, amount)"""
    base = pd.Timestamp("2022-09-01")
    return pd.DataFrame({
        "Timestamp": [(base + pd.Timedelta(minutes=m)).strftime("%Y/%m/%d %H:%M") for m, *_ in rows],
        "Account": [r[1] for r in rows], "Account.1": [r[2] for r in rows],
        "Amount Paid": [float(r[3]) for r in rows], "Payment Format": ["ACH"] * len(rows),
    })


def _graph(df):
    return build_past_only_graph_features(df, df["Account"].to_numpy(), df["Account.1"].to_numpy())[0]


def _col(X, name):
    return X[:, GRAPH_FEATURE_NAMES.index(name)]


def test_three_cycle_is_counted_only_when_the_earlier_legs_exist():
    # A->B, B->C, then C->A closes the cycle A->B->C->A (witness: B)
    src, dst, t = np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([10, 20, 30])
    assert three_cycle_witnesses(src, dst, t).tolist() == [0, 0, 1]
    # the legs arrive in the same minute as the closing transaction: they are not "earlier"
    assert three_cycle_witnesses(src, dst, np.array([10, 20, 20])).tolist() == [0, 0, 0]


def test_distinct_counterparties_and_reverse_pair_history():
    df = _tx([(0, 1, 2, 5), (10, 1, 3, 5), (20, 1, 2, 5), (30, 2, 1, 5), (90, 1, 4, 5)])
    X = _graph(df)
    # rows: 1->2, 1->3, 1->2, 2->1, 1->4. Before each, the sender has used 0, 1, 2, 0 (a different sender), 2 distinct
    # receivers; repeating 1->2 does not add one
    assert np.allclose(np.expm1(_col(X, "sender_out_distinct_receivers")), [0, 1, 2, 0, 2])
    # new receivers in the last hour: 1->2 and 1->3 first appear at minutes 0 and 10; by minute 90 both are older than 1h
    assert np.allclose(np.expm1(_col(X, "sender_out_new_receivers_1h")), [0, 1, 2, 0, 0])
    # row 3 is 2->1 and 1->2 happened twice earlier
    assert np.allclose(np.expm1(_col(X, "reverse_pair_prior_count")), [0, 0, 0, 2, 0])


def test_graph_features_do_not_depend_on_future_transactions():
    rng = np.random.default_rng(1)
    n = 80
    df = _tx([(int(m), int(a), int(b), float(x)) for m, a, b, x in
              zip(np.sort(rng.integers(0, 3000, n)), rng.integers(0, 8, n), rng.integers(0, 8, n),
                  rng.uniform(1, 90, n))])
    base = _graph(df)
    cut = 50
    later = df.copy()
    later.loc[cut:, "Account"] = (later.loc[cut:, "Account"] + 3) % 8
    later.loc[cut:, "Account.1"] = (later.loc[cut:, "Account.1"] + 5) % 8
    later.loc[cut:, "Amount Paid"] *= 40
    cut_minute = pd.to_datetime(df["Timestamp"]).iloc[cut]
    before = (pd.to_datetime(df["Timestamp"]) < cut_minute).to_numpy()   # rows strictly before the first changed minute
    other = _graph(later)
    assert np.allclose(base[before], other[before])


def test_graph_features_keep_input_row_order_when_rows_are_shuffled():
    rng = np.random.default_rng(2)
    n = 60
    df = _tx([(int(m), int(a), int(b), float(x)) for m, a, b, x in
              zip(rng.integers(0, 500, n), rng.integers(0, 7, n), rng.integers(0, 7, n), rng.uniform(1, 50, n))])
    base = _graph(df)
    perm = np.random.default_rng(5).permutation(n)
    shuffled = _graph(df.iloc[perm].reset_index(drop=True))
    assert np.allclose(shuffled, base[perm])
