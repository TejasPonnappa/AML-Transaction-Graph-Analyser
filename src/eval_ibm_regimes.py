"""
IBM test metrics split by volume regime, for any model that saved its test scores.

    python src/eval_ibm_regimes.py --scores outputs/ibm_gine_test_scores.npz --metrics outputs/metrics_ibm_gine.json

Uses the model's validation-tuned threshold from its metrics file. "Low-volume tail" days (volume under 1% of the
median day, laundering rate ~60%) are reported apart from normal-volume days, because the tail lifts pooled PR-AUC.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ibm_features import low_volume_tail_mask
from training_utils import binary_report

RAW_TRANS_PATH = 'data/raw/ibm_aml/HI-Small_Trans.csv'


def regime_report(test_idx, test_prob, threshold, timestamps, labels):
    tail = low_volume_tail_mask(timestamps)
    y_all = np.asarray(labels).astype(int)
    out = {}
    for name, rows in (('pooled', np.ones(len(y_all), dtype=bool)), ('normal-volume days', ~tail),
                       ('low-volume tail days', tail)):
        keep = rows[test_idx]
        y, p = y_all[test_idx][keep], test_prob[keep]
        rep = binary_report(y, p, threshold)
        out[name] = dict(
            rows=int(len(y)), positives=int(y.sum()),
            pr_auc=float(average_precision_score(y, p)) if y.sum() else float('nan'),
            roc_auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float('nan'),
            precision=rep['precision'], recall=rep['recall'], f1=rep['f1'])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scores', required=True, help='npz with test_idx and test_prob')
    parser.add_argument('--metrics', required=True, help='metrics json holding best_threshold')
    parser.add_argument('--out', default=None, help='where to write the json (default: next to the scores)')
    args = parser.parse_args()

    scores = np.load(args.scores)
    threshold = json.load(open(args.metrics))['best_threshold']
    raw = pd.read_csv(RAW_TRANS_PATH, usecols=['Timestamp', 'Is Laundering'])
    ts = pd.to_datetime(raw['Timestamp'], format='%Y/%m/%d %H:%M')
    report = regime_report(scores['test_idx'], scores['test_prob'], threshold, ts, raw['Is Laundering'].to_numpy())

    print(f"threshold {threshold:.4f} (from validation)")
    print(f"{'regime':24s} {'rows':>9s} {'pos':>6s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'prec':>7s} {'recall':>7s} {'F1':>7s}")
    for name, r in report.items():
        print(f"{name:24s} {r['rows']:9,d} {r['positives']:6,d} {r['pr_auc']:8.3f} {r['roc_auc']:8.3f} "
              f"{r['precision']:7.3f} {r['recall']:7.3f} {r['f1']:7.3f}")
    out = args.out or args.scores.replace('.npz', '_regimes.json')
    with open(out, 'w') as f:
        json.dump(dict(threshold=threshold, regimes=report), f, indent=2)
    print(f"written to {out}")


if __name__ == '__main__':
    main()
