import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from explain_ibm import (account_workload, alerts_per_day_for_recall, export_tree, forest_votes, scheme_recall,
                         surrogate_fidelity)


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    alert = X[:, 0] > 1.0
    y = (alert & (rng.random(n) < 0.5)).astype(int)
    return X, alert, y


def test_exported_tree_counts_are_consistent_and_leaves_carry_the_decision():
    X, alert, y = _data()
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, alert)
    tree = export_tree(clf, X, alert, y, ["f0", "f1", "f2"])
    nodes = tree["nodes"]
    assert nodes[0]["n"] == len(X) and nodes[0]["alert_frac"] == alert.mean()
    for node in nodes:
        if not node["leaf"]:
            left, right = nodes[node["left"]], nodes[node["right"]]
            assert left["n"] + right["n"] == node["n"]
    assert any(n["leaf"] and n["predict_alert"] for n in nodes) and tree["max_depth"] <= 3


def test_surrogate_fidelity_is_perfect_when_the_tree_can_represent_the_rule():
    X, alert, y = _data()
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, alert)
    fid = surrogate_fidelity(clf, X, alert, y, np.ones(len(X), bool), np.ones(len(X), bool))
    assert fid["pooled"]["agreement"] == 1.0 and fid["pooled"]["alert_recall"] == 1.0


def test_forest_votes_average_to_the_forest_probability():
    X, alert, y = _data()
    model = RandomForestClassifier(n_estimators=15, random_state=0).fit(X, alert)
    p = forest_votes(model, X[:50])
    assert p.shape == (50, 15)
    assert np.allclose(p.mean(axis=1), model.predict_proba(X[:50])[:, 1], atol=1e-6)


def test_account_workload_ranks_accounts_by_their_highest_score():
    #  accounts 0..5; account 1 and 2 are laundering (rows 0 and 1); account 3 gets a high score but is clean
    src = np.array([0, 2, 3, 4]); dst = np.array([1, 5, 5, 5])
    score = np.array([0.9, 0.8, 0.95, 0.1]); y = np.array([1, 1, 0, 0])
    res = account_workload(src, dst, score, y, np.ones(4, bool), n_accounts=6, fractions=(0.5,))
    # accounts present: 0,1,2,3,4,5 -> best scores 0.9,0.9,0.8,0.95,0.1,0.95; the top 3 hold account 3, 5 and 0 or 1
    assert res["accounts"] == 6 and res["laundering_accounts"] == 4 and res["rows"][0]["accounts_reviewed"] == 3


def test_alerts_per_day_for_recall_and_scheme_recall():
    score = np.array([0.9, 0.8, 0.7, 0.2, 0.1]); y = np.array([1, 0, 1, 0, 1])
    got = alerts_per_day_for_recall(score, y, days=2.0, recalls=(0.5,))
    assert got["0.5"]["alerts"] == 3 and got["0.5"]["alerts_per_day"] == 1.5     # need 2 of 3 positives: ranks 1 and 3
    scheme = np.array(["CYCLE", "CYCLE", "", "", "FAN-OUT"], dtype=object)
    rec = scheme_recall(scheme, score, 0.75, np.ones(5, bool))
    assert rec["CYCLE"]["recall"] == 1.0 and rec["FAN-OUT"]["recall"] == 0.0
