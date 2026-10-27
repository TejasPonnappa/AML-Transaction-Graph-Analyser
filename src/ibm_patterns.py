"""
Benchmark scheme labels for the IBM AML data.

The dataset ships HI-Small_Patterns.txt: laundering "attempts", each a block of transactions tagged with the scheme the
generator used (fan-out, cycle, ...). This module attaches that tag to the matching rows of the transactions file so
detectors and the model can be scored per scheme. The labels come from the benchmark generator; they are ground truth
for evaluation, not something the monitoring system sees.
"""
import re

import numpy as np
import pandas as pd

SCHEMES = ['FAN-OUT', 'FAN-IN', 'CYCLE', 'GATHER-SCATTER', 'SCATTER-GATHER', 'STACK', 'BIPARTITE', 'RANDOM']
UNATTRIBUTED = 'UNATTRIBUTED'      # laundering rows that no attempt block in the pattern file lists
PATTERN_FIELDS = ['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency',
                  'Amount Paid', 'Payment Currency', 'Payment Format', 'Is Laundering']
_BEGIN = re.compile(r'^BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)')


def parse_patterns(path):
    """One row per transaction line inside an attempt block: attempt id, scheme and the transaction fields."""
    rows, attempt, scheme = [], -1, None
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            begin = _BEGIN.match(line)
            if begin:
                attempt += 1
                scheme = begin.group(1)
                continue
            if line.startswith('END LAUNDERING ATTEMPT'):
                scheme = None
                continue
            if scheme is None:
                continue
            parts = line.split(',')
            if len(parts) == len(PATTERN_FIELDS):
                rows.append([attempt, scheme] + parts)
    frame = pd.DataFrame(rows, columns=['attempt', 'scheme'] + PATTERN_FIELDS)
    frame['Amount Paid'] = frame['Amount Paid'].astype(float)
    return frame


def label_rows(raw, patterns):
    """
    raw: the transactions frame (columns Timestamp, Account, Account.1, Amount Paid, Payment Format, Is Laundering).
    Returns (scheme array, attempt array) aligned to raw's rows. Laundering rows missing from the pattern file get
    UNATTRIBUTED; every other row gets '' and -1. A transaction listed in several attempts keeps the first one.
    """
    keys = ['Timestamp', 'Account', 'Account.1', 'Amount Paid', 'Payment Format']
    scheme = np.full(len(raw), '', dtype=object)
    attempt = np.full(len(raw), -1, dtype=np.int64)

    laundering = raw.index[raw['Is Laundering'].to_numpy() == 1]
    scheme[laundering] = UNATTRIBUTED
    sub = raw.loc[laundering, keys].copy()
    sub['Amount Paid'] = sub['Amount Paid'].astype(float).round(2)
    sub['_row'] = laundering
    pat = patterns[keys + ['scheme', 'attempt']].copy()
    pat['Amount Paid'] = pat['Amount Paid'].round(2)

    # identical transactions can repeat: number the repeats on both sides so each matches one pattern line
    for frame in (sub, pat):
        frame['_dup'] = frame.groupby(keys).cumcount()
    merged = sub.merge(pat, on=keys + ['_dup'], how='inner', suffixes=('', '_p'))
    merged = merged.drop_duplicates('_row')
    scheme[merged['_row'].to_numpy()] = merged['scheme'].to_numpy()
    attempt[merged['_row'].to_numpy()] = merged['attempt'].to_numpy()
    return scheme, attempt
