"""
Builds the graphs the dashboard draws, from the cached transaction table:
  hero_graph  the largest connected clusters among the current alerts
  ego_graph   one account, its counterparties and their counterparties, with cycle and pass-through edges marked
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ui.explain import BADGES

DAY_NS = 86_400 * 10 ** 9
HERO_MAX_NODES = 100


@st.cache_resource(show_spinner=False)
def load_arrays(project, mtimes, _df):
    """Plain numpy views of the transaction table (cached once; the DataFrame itself is not re-hashed)."""
    return dict(
        s=_df['SENDER_ACCOUNT_ID'].cat.codes.to_numpy(), r=_df['RECEIVER_ACCOUNT_ID'].cat.codes.to_numpy(),
        score=_df['SUSPICION_SCORE'].to_numpy(dtype=np.float32), y=_df['IS_FRAUD'].to_numpy(dtype=np.int8),
        amount=_df['TX_AMOUNT'].to_numpy(dtype=np.float64),
        t=_df['TIMESTAMP'].to_numpy('datetime64[ns]').astype('int64') if 'TIMESTAMP' in _df.columns else None,
        names=pd.Index(_df['SENDER_ACCOUNT_ID'].cat.categories),
        split=_df['SPLIT'].cat.codes.to_numpy() if 'SPLIT' in _df.columns else None,
        split_names=list(_df['SPLIT'].cat.categories) if 'SPLIT' in _df.columns else [],
        tail=_df['TAIL'].to_numpy() if 'TAIL' in _df.columns else None,
        tx_type=_df['TX_TYPE'].cat.codes.to_numpy(), tx_type_names=list(_df['TX_TYPE'].cat.categories))


def _edge_width(amount):
    return float(1.0 + min(4.0, max(0.0, np.log10(amount + 1.0) - 2.0) * 0.9))


def _badge_map(detail, row_ids, s, r, active=None):
    """account code -> set of badge labels, from detector flags on the given rows (detail rows only)."""
    out = {}
    if detail is None or len(row_ids) == 0:
        return out
    sub = detail.loc[detail.index.intersection(row_ids)]
    if sub.empty:
        return out
    pos = {int(rid): i for i, rid in enumerate(row_ids)}
    for rid, row in sub.iterrows():
        i = pos[int(rid)]
        for key, label in BADGES.items():
            if (active is not None and key not in active) or not bool(row.get(f'det_{key}', False)):
                continue
            who = (s[i],) if key in ('fan_out', 'pass_through') else ((r[i],) if key == 'fan_in' else (s[i], r[i]))
            for code in who:
                out.setdefault(int(code), set()).add(label)
    return out


def hero_graph(arr, view_idx, threshold, detail, max_nodes=HERO_MAX_NODES, active=None):
    """Largest connected clusters among the alert transactions in `view_idx`. Returns (nodes, edges, info) or None."""
    alert = view_idx[arr['score'][view_idx] >= threshold]
    if len(alert) == 0:
        return None
    s, r, sc = arr['s'][alert], arr['r'][alert], arr['score'][alert]
    codes, inv = np.unique(np.concatenate([s, r]), return_inverse=True)
    si, ri = inv[:len(s)], inv[len(s):]
    n = len(codes)
    node_score = np.zeros(n)
    np.maximum.at(node_score, si, sc)
    np.maximum.at(node_score, ri, sc)
    ncomp, labels = connected_components(coo_matrix((np.ones(len(s)), (si, ri)), shape=(n, n)), directed=False)
    size = np.bincount(labels, minlength=ncomp)
    best = np.zeros(ncomp)
    np.maximum.at(best, labels, node_score)
    keep, total = [], 0
    for comp in np.lexsort((-best, -size)):              # biggest clusters first, ties by highest score
        if total and total + size[comp] > max_nodes:
            continue
        keep.append(comp)
        total += int(size[comp])
        if total >= max_nodes:
            break
    in_keep = np.isin(labels, keep)
    node_ids = np.flatnonzero(in_keep)
    badges = _badge_map(detail, alert, s, r, active)
    names = arr['names']
    y_any = np.zeros(n, dtype=bool)
    flagged = arr['y'][alert] == 1
    y_any[si[flagged]] = True
    y_any[ri[flagged]] = True
    counts = np.bincount(np.concatenate([si, ri]), minlength=n)
    nodes = []
    for i in node_ids:
        code = int(codes[i])
        b = sorted(badges.get(code, []))
        nodes.append(dict(
            id=str(names[code]), label=str(names[code]), score=float(node_score[i]), alert=True,
            size=float(14 + 20 * node_score[i]), center=False, labelled=bool(y_any[i]), badges=b,
            tip=f"Account {names[code]}\nHighest model score {node_score[i]:.3f}\n{int(counts[i])} alert transactions"
                + (f"\nPatterns: {', '.join(b)}" if b else "") + "\nClick to investigate"))
    edge_frame = pd.DataFrame({'s': si, 'r': ri, 'score': sc, 'amount': arr['amount'][alert]})
    edge_frame = edge_frame[in_keep[edge_frame['s']]].groupby(['s', 'r']).agg(
        score=('score', 'max'), n=('score', 'size'), amount=('amount', 'sum')).reset_index()
    edges = [dict(id=f"{names[codes[e.s]]}>{names[codes[e.r]]}", source=str(names[codes[e.s]]), target=str(names[codes[e.r]]),
                  score=float(e.score), alert=True, kind='normal', w=_edge_width(e.amount),
                  tip=f"Model score {e.score:.3f}\n{int(e.n)} transaction(s), total {e.amount:,.0f}")
             for e in edge_frame.itertuples()]
    info = dict(alerts=int(len(alert)), clusters=len(keep), accounts=len(nodes), total_alert_accounts=int(n))
    return nodes, edges, info


def ego_graph(arr, detail, account, threshold, max_neighbors=26, max_second=10, per_second=4, active=None):
    """One account with its counterparties and theirs. Returns (nodes, edges, info) or None if the account is unknown."""
    names = arr['names']
    if account not in names:
        return None
    a = int(names.get_loc(account))
    s, r, sc, amt, t = arr['s'], arr['r'], arr['score'], arr['amount'], arr['t']
    rows1 = np.flatnonzero((s == a) | (r == a))
    if len(rows1) == 0:
        return None
    out1 = s[rows1] == a
    cp1 = np.where(out1, r[rows1], s[rows1])
    keep_self = cp1 != a
    first = pd.DataFrame({'cp': cp1[keep_self], 'out': out1[keep_self], 'score': sc[rows1][keep_self],
                          'amt': amt[rows1][keep_self], 'row': rows1[keep_self]})
    ego_info = dict(transactions=int(len(rows1)), alerts=int((sc[rows1] >= threshold).sum()))
    if first.empty:
        return [], [], ego_info
    by_cp = first.groupby('cp')['score'].max().nlargest(max_neighbors)
    neighbours = set(int(c) for c in by_cp.index)
    agg1 = first[first['cp'].isin(neighbours)].groupby(['cp', 'out']).agg(
        score=('score', 'max'), n=('row', 'size'), amount=('amt', 'sum')).reset_index()

    edges = {}                                                       # (from, to) -> dict
    def add_edge(u, v, score, n, amount):
        cur = edges.get((u, v))
        if cur is None:
            edges[(u, v)] = dict(score=float(score), n=int(n), amount=float(amount))
        else:
            cur.update(score=max(cur['score'], float(score)), n=cur['n'] + int(n), amount=cur['amount'] + float(amount))
    for e in agg1.itertuples():
        add_edge(*((a, int(e.cp)) if e.out else (int(e.cp), a)), e.score, e.n, e.amount)

    node_score = {a: float(sc[rows1].max())}
    node_count = {a: int(len(rows1))}
    for c in neighbours:
        sel = first[first['cp'] == c]
        node_score[c], node_count[c] = float(sel['score'].max()), int(len(sel))

    second_nodes = set()
    for b in by_cp.index[:max_second]:
        b = int(b)
        rows2 = np.flatnonzero((s == b) | (r == b))
        out2 = s[rows2] == b
        cp2 = np.where(out2, r[rows2], s[rows2])
        ok = (cp2 != a) & (cp2 != b)
        if not ok.any():
            continue
        second = pd.DataFrame({'cp': cp2[ok], 'out': out2[ok], 'score': sc[rows2][ok], 'amt': amt[rows2][ok], 'row': rows2[ok]})
        ranked = second.groupby('cp')['score'].max()
        chosen = set(int(c) for c in ranked.nlargest(per_second).index) | (set(int(c) for c in ranked.index) & neighbours)
        for e in second[second['cp'].isin(chosen)].groupby(['cp', 'out']).agg(
                score=('score', 'max'), n=('row', 'size'), amount=('amt', 'sum')).reset_index().itertuples():
            add_edge(*((b, int(e.cp)) if e.out else (int(e.cp), b)), e.score, e.n, e.amount)
            if int(e.cp) not in neighbours:
                second_nodes.add(int(e.cp))
                node_score[int(e.cp)] = max(node_score.get(int(e.cp), 0.0), float(e.score))
                node_count[int(e.cp)] = node_count.get(int(e.cp), 0) + int(e.n)

    # patterns around the searched account
    kind = {}
    adjacency = {}
    for (u, v) in edges:
        adjacency.setdefault(u, set()).add(v)
    cycles = 0
    for x in adjacency.get(a, ()):
        for y in adjacency.get(x, ()):
            if y != a and a in adjacency.get(y, ()):
                for pair in ((a, x), (x, y), (y, a)):
                    kind[pair] = 'cycle'
                cycles += 1
    passes = 0
    if t is not None:
        inc = first[~first['out']].nlargest(400, 'amt')
        outg = first[first['out']].nlargest(400, 'amt')
        if len(inc) and len(outg):
            dt = t[outg['row'].to_numpy()][None, :] - t[inc['row'].to_numpy()][:, None]
            ratio = outg['amt'].to_numpy()[None, :] / np.maximum(inc['amt'].to_numpy()[:, None], 1e-9)
            ii, jj = np.nonzero((dt >= 0) & (dt <= DAY_NS) & (ratio >= 0.7) & (ratio <= 1.1))
            for i, j in zip(ii, jj):
                u, w = int(inc['cp'].to_numpy()[i]), int(outg['cp'].to_numpy()[j])
                if (u, a) in edges and (a, w) in edges:
                    kind.setdefault((u, a), 'pass')
                    kind.setdefault((a, w), 'pass')
                    passes += 1

    badge_rows = rows1
    badges = _badge_map(detail, badge_rows, s[badge_rows], r[badge_rows], active)
    y_rows = rows1[arr['y'][rows1] == 1]
    labelled = set(int(x) for x in np.concatenate([s[y_rows], r[y_rows]])) if len(y_rows) else set()
    schemes = []
    if detail is not None and len(y_rows):
        schemes = sorted(set(str(x) for x in detail.loc[detail.index.intersection(y_rows), 'scheme'] if str(x)))

    all_nodes = [a] + sorted(neighbours) + sorted(second_nodes - neighbours)
    nodes = []
    for c in all_nodes:
        b = sorted(badges.get(c, []))
        score = node_score.get(c, 0.0)
        nodes.append(dict(
            id=str(names[c]), label=str(names[c]), score=score, alert=bool(score >= threshold),
            size=float(34 if c == a else 13 + 20 * score), center=bool(c == a), labelled=bool(c in labelled), badges=b,
            tip=f"Account {names[c]}\nHighest model score {score:.3f}\n{node_count.get(c, 0)} transaction(s) shown"
                + (f"\nPatterns: {', '.join(b)}" if b else "")))
    edge_list = []
    for (u, v), e in edges.items():
        k = kind.get((u, v), 'normal')
        extra = {'cycle': '\nCloses a cycle through the searched account', 'pass': '\nPass-through: money in, then out within a day'}.get(k, '')
        edge_list.append(dict(id=f"{names[u]}>{names[v]}", source=str(names[u]), target=str(names[v]), score=e['score'],
                              alert=bool(e['score'] >= threshold), kind=k, w=_edge_width(e['amount']),
                              tip=f"Model score {e['score']:.3f}\n{e['n']} transaction(s), total {e['amount']:,.0f}{extra}"))
    ego_info.update(neighbours=len(neighbours), second_hop=len(second_nodes - neighbours), cycles=cycles,
                    pass_through_pairs=passes, benchmark_schemes=schemes, badges=sorted(badges.get(a, [])))
    return nodes, edge_list, ego_info
