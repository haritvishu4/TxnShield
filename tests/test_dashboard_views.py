import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.components import filter_history, history_frame, html, recent_amount_events

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Overview", "Transaction Analysis", "Monitoring & Audits", "Model Performance", "Architecture", "System Information"]


def records():
    return [
        {"id": i, "transaction_id": f"TXN-DEMO-{i}", "timestamp": f"2026-09-04T12:0{i}:00",
         "amount": amount, "fraud_probability": score / 100, "risk_score": score,
         "risk_level": tier, "is_fraud": score >= 41.59, "decision": "Review" if score >= 30 else "Approve",
         "latency_ms": 8.5, "model_version": "1.0.0"}
        for i, (amount, score, tier) in enumerate([(6.42, .11, "Low Risk"), (150.66, 46.16, "Medium Risk"), (727.91, 96.96, "Critical Risk")], start=1)
    ]


@pytest.fixture
def dashboard_api(monkeypatch):
    st.cache_data.clear()
    state = {"history": records(), "offline": False}
    saved_metrics = json.loads((ROOT / "models" / "metrics.json").read_text())

    def get(url, **kwargs):
        if state["offline"]:
            raise requests.ConnectionError("Test API offline")
        if url.endswith("/health"):
            data = {"model_loaded": True, "status": "healthy", "model_family": "RandomForestClassifier", "version": "1.0.0", "optimal_threshold": .4159}
        elif "/history" in url:
            data = state["history"]
        else:
            data = saved_metrics
        return Mock(status_code=200, json=lambda: data)

    monkeypatch.setenv("API_BASE_URL", "http://dashboard-views.invalid")
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", Mock(side_effect=AssertionError("Unexpected prediction")))
    monkeypatch.setattr(requests, "delete", Mock(side_effect=AssertionError("Unexpected history deletion")))
    yield state
    st.cache_data.clear()


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("with_records", [False, True])
def test_every_page_renders_without_mutating_audits(dashboard_api, page, with_records):
    if not with_records:
        dashboard_api["history"] = []
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=15)
    app.session_state["page"] = page
    app.run()
    assert not app.exception
    assert not app.error


def test_offline_status_is_not_presented_as_zero_transactions(dashboard_api):
    dashboard_api["offline"] = True
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py")).run()
    assert not app.exception
    content = " ".join(element.value for element in app.markdown)
    assert "Service unavailable" in content
    assert "Audit service unavailable" in content
    assert "Model online" not in content


def test_reset_requires_confirmation_and_filter_changes_are_read_only(dashboard_api):
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"))
    app.session_state["page"] = "Monitoring & Audits"
    app.run()
    app.text_input(key="audit_search").set_value("demo-2").run()
    assert not app.exception
    assert any("1 of 3 loaded records" in item.value for item in app.caption)
    next(button for button in app.button if button.label == "Clear Demo Audit History").click().run()
    assert not app.exception
    assert any("Confirm the checkbox" in item.value for item in app.info)


def test_filters_are_literal_case_insensitive_and_composable():
    frame = history_frame(records())
    assert len(filter_history(frame, "DEMO")) == 3
    assert filter_history(frame, "[.*").empty
    result = filter_history(frame, "demo", ["Critical Risk"], True)
    assert result["transaction_id"].tolist() == ["TXN-DEMO-3"]
    assert str(frame["timestamp"].dt.tz) == "UTC"
    assert html('<script>alert("x")</script>') == '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;'


def test_amount_chart_keeps_repeat_assessments_distinct_and_bounded():
    rows = records()
    rows[2]["transaction_id"] = rows[1]["transaction_id"]
    frame = history_frame(list(reversed(rows)))
    events = recent_amount_events(frame, limit=2)
    assert events["Audit event"].tolist() == ["2", "3"]
    assert events["transaction_id"].tolist() == ["TXN-DEMO-2", "TXN-DEMO-2"]
    assert events["amount"].tolist() == [150.66, 727.91]
    assert events["Prediction"].tolist() == ["Flagged", "Flagged"]
    assert frame["id"].tolist() == [3, 2, 1]
    assert "Audit event" not in frame
    assert recent_amount_events(history_frame([])).empty


def test_ui_with_real_artifacts_matches_temporary_sqlite(monkeypatch, tmp_path):
    """Exercise UI → unchanged API → real model → isolated audit database."""
    from fastapi.testclient import TestClient
    from api.app import app as api_app
    from src.database.connection import get_session_direct
    from src.database.models import TransactionAudit

    st.cache_data.clear()
    monkeypatch.setenv("FRAUD_DATABASE_URL", f"sqlite:///{tmp_path / 'ui-integration.db'}")
    monkeypatch.setenv("API_BASE_URL", "http://isolated-ui.invalid")
    with TestClient(api_app) as client:
        monkeypatch.setattr(requests, "get", lambda url, **kwargs: client.get(url.replace("http://isolated-ui.invalid", "")))
        monkeypatch.setattr(requests, "post", lambda url, **kwargs: client.post(url.replace("http://isolated-ui.invalid", ""), json=kwargs["json"]))
        monkeypatch.setattr(requests, "delete", Mock(side_effect=AssertionError("Unexpected delete")))
        ui = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30).run()
        ui.radio(key="page").set_value("Transaction Analysis").run()
        for key, expected_probability, expected_score in [("coffee", .0011, .11), ("electronics", .4616, 46.16), ("fraud", .9696, 96.96), ("electronics", .4616, 46.16)]:
            ui.button(key=f"preset_{key}").click().run()
            assert not ui.exception
            assert not ui.error
            result = ui.session_state["last_prediction"]
            assert result["fraud_probability"] == expected_probability
            assert result["risk_score"] == expected_score
            with get_session_direct() as session:
                row = session.query(TransactionAudit).order_by(TransactionAudit.id.desc()).first()
                for field in ("fraud_probability", "risk_score", "risk_level", "is_fraud", "decision", "model_version", "latency_ms"):
                    assert getattr(row, field) == result[field]
                assert json.loads(row.features_json) == {k: v for k, v in ui.session_state["last_payload"].items() if k != "transaction_id"}
        with get_session_direct() as session:
            assert session.query(TransactionAudit).count() == 4
    st.cache_data.clear()
