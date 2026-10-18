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
