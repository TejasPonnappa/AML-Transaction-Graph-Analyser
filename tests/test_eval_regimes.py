import numpy as np
import pandas as pd

from eval_ibm_regimes import regime_report


def test_regime_report_separates_the_low_volume_tail():
    days = ["2022/09/01 10:00"] * 200 + ["2022/09/02 10:00"] * 200 + ["2022/09/03 10:00"]
    ts = pd.to_datetime(pd.Series(days), format='%Y/%m/%d %H:%M')
    labels = np.array([0] * 396 + [1] * 4 + [1])
    prob = np.where(labels == 1, 0.9, 0.1)
    rep = regime_report(np.arange(401), prob, 0.5, ts, labels)
    assert rep['pooled']['rows'] == 401 and rep['low-volume tail days']['rows'] == 1
    assert rep['low-volume tail days']['positives'] == 1 and rep['normal-volume days']['positives'] == 4
