"""
Training loop and evaluation shared by train.py, train_ibm.py and train_bitcoin.py.

Selection rules (the same for every dataset):
  * the checkpoint of each run is chosen on VALIDATION average precision (PR-AUC), not ROC-AUC or F1 at 0.5;
  * the decision threshold is the F1-maximising point of the VALIDATION precision-recall curve (no fixed grid),
    applied to test once with `>=`, the same comparison predict*.py uses;
  * hyperparameters such as the positive-class weight must be chosen on validation numbers only;
  * several seeds can be trained; the seed with the best validation PR-AUC is the one saved and used for
    predictions, and the spread over all seeds is reported alongside.
Test data is touched only for the final report.
"""
import copy
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tune_threshold(y, p):
    """Threshold maximising F1 on the precision-recall curve; predictions use `p >= threshold`."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5


def safe_average_precision(y, p):
    return float(average_precision_score(y, p)) if (y == 1).any() else 0.0


def safe_roc_auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < (y == 1).sum() < len(y) else float('nan')


def binary_report(y, p, threshold):
    """Metrics at `threshold` (`>=`). The confusion matrix is always 2x2, even if a class is absent."""
    pred = p >= threshold
    pos = y == 1
    tp, fp = int((pred & pos).sum()), int((pred & ~pos).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & ~pos).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1,
                confusion_matrix=dict(tp=tp, fp=fp, fn=fn, tn=tn))


def _state_copy(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_training(*, build_model, loss_fn, outputs_fn, data, y, masks, device, seeds, num_epochs, lr, patience,
                 log=print, max_seconds=None):
    """
    build_model()            -> fresh nn.Module
    loss_fn(raw, target)     -> scalar loss on the selected rows
    outputs_fn(model, data, idx) -> (raw outputs, positive-class probability) for the rows in idx only
    y                        -> targets over all rows (tensor on `device`)
    masks                    -> dict of boolean tensors: 'train', 'val', 'test'

    Returns {'runs': [...], 'best': index of the run with the best validation PR-AUC}. Each run holds
    the selected state dict, history, validation/test probabilities and metrics. `max_seconds` caps the wall-clock
    time of each seed's training loop; the best checkpoint so far is kept when it is hit.
    """
    train_idx, val_idx, test_idx = (torch.where(masks[k])[0] for k in ('train', 'val', 'test'))
    y_np = y.detach().cpu().numpy()
    runs = []
    for seed in seeds:
        set_seed(seed)
        model = build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_pr_auc': []}
        best_ap, best_state, wait = -1.0, None, 0
        started = time.time()
        stopped_by_time = False
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()
            raw, _ = outputs_fn(model, data, train_idx)
            loss = loss_fn(raw, y[train_idx])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                raw_v, prob_v = outputs_fn(model, data, val_idx)
                val_loss = loss_fn(raw_v, y[val_idx]).item()
                val_ap = safe_average_precision(y_np[val_idx.cpu().numpy()], prob_v.cpu().numpy())
            scheduler.step(val_ap)
            for k, v in zip(history, (epoch, loss.item(), val_loss, val_ap)):
                history[k].append(v)

            improved = val_ap > best_ap
            if improved:
                best_ap, best_state, wait = val_ap, _state_copy(model), 0
            else:
                wait += 1
            if epoch % 10 == 0 or improved:
                log(f"seed {seed} epoch {epoch:03d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | "
                    f"val PR-AUC {val_ap:.4f}{'  <- best' if improved else ''}")
            if wait >= patience:
                log(f"seed {seed}: early stop at epoch {epoch} (no validation PR-AUC gain in {patience} epochs)")
                break
            if max_seconds and time.time() - started > max_seconds:
                log(f"seed {seed}: stopped at epoch {epoch}: time limit of {max_seconds / 60:.0f} min reached")
                stopped_by_time = True
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pv = outputs_fn(model, data, val_idx)[1].cpu().numpy()
            pt = outputs_fn(model, data, test_idx)[1].cpu().numpy()
        yv, yt = y_np[val_idx.cpu().numpy()], y_np[test_idx.cpu().numpy()]
        threshold = tune_threshold(yv, pv)
        test = binary_report(yt, pt, threshold)
        test.update(pr_auc=safe_average_precision(yt, pt), roc_auc=safe_roc_auc(yt, pt),
                    f1_at_0_5=binary_report(yt, pt, 0.5)['f1'], threshold=threshold)
        runs.append(dict(seed=seed, val_pr_auc=best_ap, state=best_state, history=history, test=test,
                         epochs_run=len(history['epoch']), stopped_by_time=stopped_by_time,
                         train_seconds=time.time() - started,
                         val_idx=val_idx.cpu().numpy(), val_prob=pv, test_idx=test_idx.cpu().numpy(), test_prob=pt))
        log(f"seed {seed}: val PR-AUC {best_ap:.4f} | test PR-AUC {test['pr_auc']:.4f} | threshold {threshold:.4f} "
            f"| precision {test['precision']:.3f} recall {test['recall']:.3f} F1 {test['f1']:.3f}")
    best = int(np.argmax([r['val_pr_auc'] for r in runs]))
    return dict(runs=runs, best=best)


def summarise_runs(result, extra=None):
    """Metrics JSON payload: headline numbers come from the validation-selected run, with spread over seeds."""
    runs = result['runs']
    best = runs[result['best']]
    keys = ('pr_auc', 'roc_auc', 'precision', 'recall', 'f1')
    spread = {k: dict(mean=float(np.nanmean([r['test'][k] for r in runs])),
                      std=float(np.nanstd([r['test'][k] for r in runs]))) for k in keys}
    payload = {
        'test_roc_auc': best['test']['roc_auc'],
        'test_pr_auc': best['test']['pr_auc'],
        'test_precision': best['test']['precision'],
        'test_recall': best['test']['recall'],
        'test_f1_default_thresh': best['test']['f1_at_0_5'],
        'test_f1_tuned_thresh': best['test']['f1'],
        'test_f1': best['test']['f1'],
        'confusion_matrix': best['test']['confusion_matrix'],
        'best_threshold': best['test']['threshold'],
        'selection': 'validation PR-AUC; threshold from the validation precision-recall curve (F1)',
        'selected_seed': best['seed'],
        'seeds': [dict(seed=r['seed'], val_pr_auc=r['val_pr_auc'], test_pr_auc=r['test']['pr_auc'],
                       test_f1=r['test']['f1']) for r in runs],
        'seed_spread_test': spread,
    }
    payload.update(extra or {})
    return payload
