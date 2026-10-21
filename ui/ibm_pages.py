from ui.data_store import load_edges
from ui.edge_dashboard import render_edge_project
from ui.metrics_loader import load_metrics

METRICS_PATH = 'outputs/metrics_ibm_rf.json'


def draw_ibm_router():
    render_edge_project(dict(
        key='ibm',
        title="IBM AML Transactions",
        subtitle="Edge classification on the IBM HI-Small synthetic banking dataset. A RandomForest on past-only "
                 "transaction history and graph-structure features scores each transaction for money-laundering "
                 "activity.",
        loader=load_edges,
        metrics=lambda: load_metrics(METRICS_PATH),
        metrics_path=METRICS_PATH,
        predict_cmd='python src/predict_ibm_rf.py',
        time_kind='datetime',
        currency='localized',
        amount_note="Mixed currencies, as paid",
    ))
