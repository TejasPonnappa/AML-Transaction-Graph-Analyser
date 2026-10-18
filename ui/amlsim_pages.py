from ui.data_store import load_edges
from ui.edge_dashboard import render_edge_project
from ui.metrics_loader import load_metrics

METRICS_PATH = 'outputs/metrics.json'


def draw_amlsim_router():
    render_edge_project(dict(
        key='amlsim',
        title="AMLSim Transactions",
        subtitle="Edge classification on simulated banking transfers. Each transaction is scored for "
                 "money-laundering activity.",
        loader=load_edges,
        metrics=lambda: load_metrics(METRICS_PATH),
        metrics_path=METRICS_PATH,
        predict_cmd='python src/predict.py',
        time_kind='step',
        currency='dollar',
        amount_note="Simulated currency units",
    ))
