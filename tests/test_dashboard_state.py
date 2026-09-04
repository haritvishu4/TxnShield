import ast
from pathlib import Path
from unittest.mock import Mock

import requests
from streamlit.testing.v1 import AppTest

from dashboard.state import sync_preset_state


def test_switching_presets_replaces_every_widget_value():
    presets = {
        "coffee": {"name": "Coffee", "features": {"Time": 10.0, "Amount": 6.42, "V1": 1.0}},
        "electronics": {"name": "Electronics", "features": {"Time": 20.0, "Amount": 150.66, "V1": 2.0}},
    }
    state = {}

    sync_preset_state(state, "coffee", presets)
    sync_preset_state(state, "electronics", presets)

    assert state["txn_features"] == presets["electronics"]["features"]
    assert state["input_Time"] == 20.0
    assert state["input_Amount"] == 150.66
    assert state["input_V1"] == 2.0
    assert state["input_transaction_id"] == "TXN-PRESET-ELECTRONICS"
    assert state["auto_trigger"] is True


def test_real_dashboard_preset_buttons_send_complete_vectors(monkeypatch):
    """Exercise keyed widgets across reruns without contacting a real API."""
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    module = ast.parse(app_path.read_text(encoding="utf-8"))
    presets = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DATASET_PRESETS"
                for target in node.targets)
    )
    sent_payloads = []

    def fake_get(url, **kwargs):
        if url.endswith("/health"):
            data = {"model_loaded": True, "model_family": "RandomForest", "optimal_threshold": 0.4159}
        elif "/history" in url:
            data = []
        else:
            data = {}
        return Mock(status_code=200, json=lambda: data)

    def fake_post(url, *, json, **kwargs):
        assert url == "http://review-test.invalid/predict"
        sent_payloads.append(json.copy())
        result = {
            "fraud_probability": 0.4616, "risk_score": 46.16,
            "risk_level": "Medium Risk", "latency_ms": 1.0,
            "decision": "Step-Up Authentication", "top_risk_drivers": [],
        }
        return Mock(status_code=200, json=lambda: result)

    monkeypatch.setenv("API_BASE_URL", "http://review-test.invalid/")
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "delete", Mock(side_effect=AssertionError("Unexpected delete")))
    app = AppTest.from_file(str(app_path), default_timeout=15).run()
    assert not app.exception
    assert not sent_payloads  # Opening Overview must not create audit rows.
    app.radio(key="page").set_value("Transaction Analysis").run()
    assert not app.exception

    for key in ("coffee", "electronics", "fraud", "coffee", "electronics"):
        button_text = {"coffee": "Coffee Shop", "electronics": "Electronics Store", "fraud": "High-Risk Attack"}[key]
        next(button for button in app.button if button_text in button.label).click().run()
        assert not app.exception
        expected = presets[key]["features"]
        assert len(expected) == 30
        assert sent_payloads[-1] == {"transaction_id": f"TXN-PRESET-{key.upper()}", **expected}
        for feature, value in expected.items():
            assert app.number_input(key=f"input_{feature}").value == value

    calls_before_navigation = len(sent_payloads)
    app.radio(key="page").set_value("Overview").run()
    app.radio(key="page").set_value("Transaction Analysis").run()
    assert not app.exception
    assert len(sent_payloads) == calls_before_navigation
    for feature, value in presets["electronics"]["features"].items():
        assert app.number_input(key=f"input_{feature}").value == value

    app.number_input(key="input_Amount").set_value(299.99)
    app.number_input(key="input_V1").set_value(-1.2345)
    next(button for button in app.button if button.label == "Analyze transaction").click().run()
    assert not app.exception
    assert len(sent_payloads) == calls_before_navigation + 1
    assert sent_payloads[-1]["Amount"] == 299.99
    assert sent_payloads[-1]["V1"] == -1.2345
