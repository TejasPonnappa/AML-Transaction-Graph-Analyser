import json

import pytest

from conftest import make_graph

CASES = [
    ("predict_ibm", "data/processed/graph_data_ibm.pt"),
    ("predict_bitcoin", "data/processed/graph_data_bitcoin.pt"),
]


@pytest.fixture(params=CASES, ids=[c[0] for c in CASES])
def mod(request, load_script):
    name, path = request.param
    return load_script(name, path, make_graph())


def test_default_when_metrics_missing(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "METRICS_PATH", str(tmp_path / "nope.json"))
    assert mod.load_decision_threshold() == 0.5


def test_reads_best_threshold(mod, tmp_path, monkeypatch):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"best_threshold": 0.137}))
    monkeypatch.setattr(mod, "METRICS_PATH", str(p))
    assert mod.load_decision_threshold() == pytest.approx(0.137)


def test_default_when_key_absent(mod, tmp_path, monkeypatch):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"f1": 0.9}))
    monkeypatch.setattr(mod, "METRICS_PATH", str(p))
    assert mod.load_decision_threshold() == mod.DEFAULT_THRESHOLD


def test_ui_get_threshold():
    pytest.importorskip("streamlit")
    from ui.metrics_loader import get_threshold
    assert get_threshold({"best_threshold": "0.25"}) == 0.25
    assert isinstance(get_threshold({"best_threshold": 1}), float)
    assert get_threshold({}) == 0.5
    assert get_threshold({}, default=0.3) == 0.3


def test_ui_load_metrics(tmp_path):
    pytest.importorskip("streamlit")
    from ui.metrics_loader import load_metrics
    assert load_metrics(str(tmp_path / "missing.json")) == {}
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps({"best_threshold": 0.4}))
    assert load_metrics(str(p)) == {"best_threshold": 0.4}
