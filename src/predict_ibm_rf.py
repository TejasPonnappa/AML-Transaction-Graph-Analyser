"""
IBM AML scoring with the RandomForest model of record (past-only features).

    python src/predict_ibm_rf.py

Builds the 43 past-only features (ibm_features.py), fits a RandomForest on the TRAINING rows (all positives plus a
weighted sample of negatives), tunes the decision threshold on the full VALIDATION split (max F1 on the precision-recall
curve), and scores every transaction. Writes
  outputs/ibm_predictions_rf.csv   one row per transaction, with a SPLIT column (train / val / test)
  outputs/metrics_ibm_rf.json      test metrics at the validation threshold, pooled and per volume regime
  outputs/model_ibm_rf.joblib      the fitted forest

Scores on TRAIN rows are in-sample and optimistic; use the SPLIT column. Scores are class-weighted forest outputs between
0 and 1: they are not calibrated probabilities.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from baselines import factorize_accounts, load_masks
from eval_ibm_regimes import regime_report
from ibm_features import build_past_only_features, build_past_only_graph_features
from training_utils import binary_report, tune_threshold

RAW_TRANS_PATH = 'data/raw/ibm_aml/HI-Small_Trans.csv'
GRAPH_PATH = 'data/processed/graph_data_ibm.pt'          # only its train/val/test masks are used
OUTPUT_SCORES_PATH = 'outputs/ibm_predictions_rf.csv'
METRICS_PATH = 'outputs/metrics_ibm_rf.json'
MODEL_PATH = 'outputs/model_ibm_rf.joblib'
COLUMNS = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency',
           'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering']


def load_frame():
    df = pd.read_csv(RAW_TRANS_PATH)
    df.columns = COLUMNS      # the raw file repeats the header name 'Account'
    return df


def fit_forest(X, y, train_mask, neg_frac, seed, n_estimators):
    rng = np.random.default_rng(seed)
    train = np.flatnonzero(train_mask)
    weight = None
    if neg_frac < 1.0:
        keep = (y[train] == 1) | (rng.random(len(train)) < neg_frac)
        weight = np.where(y[train][keep] == 1, 1.0, 1.0 / neg_frac)
        train = train[keep]
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, class_weight='balanced_subsample',
                                   n_jobs=-1, random_state=seed)
    model.fit(X[train], y[train], sample_weight=weight)
    return model


def predict_in_chunks(model, X, chunk=500_000):
    return np.concatenate([model.predict_proba(X[i:i + chunk])[:, 1] for i in range(0, len(X), chunk)])


def generate_rf_predictions(neg_frac=0.1, seed=0, n_estimators=200, log=print):
    t0 = time.time()
    df = load_frame()
    n = len(df)
    log(f"1. Loaded {n:,} transactions")
    (src, dst), _ = factorize_accounts(df['Account'], df['Account.1'])
    y = df['Is Laundering'].to_numpy().astype(int)
    masks, _ = load_masks(GRAPH_PATH, n)

    log("2. Building past-only features (history + graph structure)...")
    past, past_names = build_past_only_features(df, src, dst)
    graph, graph_names = build_past_only_graph_features(df, src, dst)
    X = np.hstack([past, graph])
    names = list(past_names) + list(graph_names)
    log(f"   -> {X.shape[1]} features in {time.time() - t0:.0f}s")

    log("3. Fitting the RandomForest on training rows only...")
    model = fit_forest(X, y, masks['train'], neg_frac, seed, n_estimators)
    val, test = np.flatnonzero(masks['val']), np.flatnonzero(masks['test'])
    p_val = model.predict_proba(X[val])[:, 1]
    threshold = tune_threshold(y[val], p_val)            # validation only
    scores = predict_in_chunks(model, X)
    log(f"   -> threshold {threshold:.4f} from validation; scored all rows at {time.time() - t0:.0f}s")

    p_test = scores[test]
    rep = binary_report(y[test], p_test, threshold)
    ts = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
    metrics = {
        'model_kind': 'random_forest', 'n_features': int(X.shape[1]), 'feature_names': names,
        'neg_frac': neg_frac, 'seed': seed, 'n_estimators': n_estimators,
        'test_roc_auc': float(roc_auc_score(y[test], p_test)),
        'test_pr_auc': float(average_precision_score(y[test], p_test)),
        'test_precision': rep['precision'], 'test_recall': rep['recall'],
        'test_f1_tuned_thresh': rep['f1'], 'test_f1': rep['f1'],
        'confusion_matrix': rep['confusion_matrix'], 'best_threshold': threshold,
        'selection': 'threshold from the validation precision-recall curve (max F1); test used only for this report',
        'regimes': regime_report(test, p_test, threshold, ts, y),
    }
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(model, MODEL_PATH)

    split = np.full(n, 'unassigned', dtype=object)
    for name in ('train', 'val', 'test'):
        split[masks[name]] = name
    out = pd.DataFrame({
        'SENDER_ACCOUNT_ID': df['Account'], 'RECEIVER_ACCOUNT_ID': df['Account.1'],
        'TX_AMOUNT': df['Amount Paid'], 'TX_TYPE': df['Payment Format'], 'IS_FRAUD': y,
        'SUSPICION_SCORE': scores, 'PREDICTED_FRAUD': (scores >= threshold).astype(int), 'SPLIT': split,
    })
    out.to_csv(OUTPUT_SCORES_PATH, index=False)
    log(f"4. Wrote {OUTPUT_SCORES_PATH}, {METRICS_PATH} and {MODEL_PATH} ({time.time() - t0:.0f}s total)")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--neg-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-estimators', type=int, default=200)
    args = parser.parse_args()
    if not os.path.exists(RAW_TRANS_PATH) or not os.path.exists(GRAPH_PATH):
        print(f"Error: need {RAW_TRANS_PATH} and {GRAPH_PATH}. Run 'python src/data_preprocessing_ibm.py' first.")
    else:
        m = generate_rf_predictions(args.neg_frac, args.seed, args.n_estimators)
        print(f"Test PR-AUC {m['test_pr_auc']:.4f} | precision {m['test_precision']:.4f} | "
              f"recall {m['test_recall']:.4f} | F1 {m['test_f1']:.4f}")
