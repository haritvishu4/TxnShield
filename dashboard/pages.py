"""FinTech dashboard views. No model, scoring, or database operations live here."""
import json

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from dashboard.components import (
    RISK_BACKGROUNDS, RISK_COLORS, activity, assessed_amounts, audit_table, chart, detail,
    distribution, empty, filter_history, history_kpis, html, model_health, number, title,
)


def overview(health, metrics, frame, available):
    snapshot, workspace = st.columns([1, 3.8], gap="medium")
    with snapshot:
        history_kpis(frame, available, vertical=True)
        st.caption("AUDIT SNAPSHOT · Latest 500 records, including repeat assessments.")
    with workspace:
        amounts, trend = st.columns(2)
        with amounts, st.container(border=True):
            title("Assessed amounts", "Last 12 audit inputs · not approved payment volume")
            assessed_amounts(frame, height=225)
        with trend, st.container(border=True):
            title("Risk activity", "Recorded predictions · UTC")
            activity(frame, height=225)
        risks, model = st.columns(2)
        with risks, st.container(border=True):
            title("Risk distribution", "Business tiers · current audit window")
            distribution(frame, height=225)
        with model, st.container(border=True):
            model_health(health, metrics, frame)
    with st.container(border=True):
        title("Recent transactions", "Latest 8 audit events · demo / local prediction history")
        audit_table(frame, limit=8)
    alerts, context = st.columns([1.55, 1])
    with alerts, st.container(border=True):
        title("Priority review queue", "Recent high and critical scores · no automated account action")
        flagged = frame[frame["risk_score"] >= 70] if not frame.empty else frame
        if flagged.empty:
            empty("No high / critical alerts", "Transactions with a score of 70+ will appear here.")
        else:
            audit_table(flagged, limit=5)
    with context, st.container(border=True):
        title("A score is a signal, not a verdict", "Understand the decision before taking action")
        st.caption("The binary fraud flag uses the validation-tuned threshold. Business actions use separate 30 / 70 / 90 risk boundaries. Flagged rate here describes predictions, not confirmed fraud prevalence.")
        st.button("Analyze a transaction →", key="overview_analyze", on_click=lambda: st.session_state.update(page="Transaction Analysis"))


def render_result(health):
    result = st.session_state.get("last_prediction")
    if not result:
        with st.container(border=True):
            title("Risk assessment", "Prediction & operational recommendation")
            empty("Ready to assess", "Choose a tested preset or submit your own feature vector.")
            detail("Features", "Time, Amount + V1–V28")
            detail("Decision threshold", number(health.get("optimal_threshold")))
            st.caption("Predictions are saved by the existing API. Selecting a preset runs one assessment.")
        return
    level = result["risk_level"]
    color = RISK_COLORS.get(level, "#2855C5")
    background = RISK_BACKGROUNDS.get(level, "#EDF2FF")
    with st.container(border=True):
        title("Risk assessment", result.get("transaction_id", "Latest successful prediction"))
        st.markdown(f'<span class="risk-pill" style="color:{color};background:{background}">{html(level)}</span>'
                    f'<div class="result-score">{result["risk_score"]:.2f}<span> / 100</span></div>'
                    f'<div class="risk-meter"><div style="width:{max(0, min(100, result["risk_score"]))}%;background:{color}"></div></div>', unsafe_allow_html=True)
        detail("Fraud probability", f'{result["fraud_probability"] * 100:.2f}%')
        detail("Prediction", result.get("prediction", "Potential fraud" if result.get("is_fraud") else "Not flagged"))
        st.markdown(f'<div class="action-note"><strong>OPERATIONAL RECOMMENDATION</strong>{html(result["decision"])}</div>', unsafe_allow_html=True)
        detail("Inference latency", f'{result["latency_ms"]:.2f} ms')
        detail("Model version", result.get("model_version", "—"))
        detail("Threshold at assessment", number(st.session_state.get("prediction_threshold")))
        st.caption("Model probability is not separately calibrated. Risk tier and binary classification are different signals.")
    with st.container(border=True):
        title("Top risk drivers", "SHAP contributions relative to the model baseline")
        drivers = result.get("top_risk_drivers", [])
        if drivers:
            data = pd.DataFrame(drivers)
            figure = px.bar(data.sort_values("shap_value"), x="shap_value", y="feature", orientation="h", color="impact",
                            color_discrete_map={"Increases Risk": "#D75A68", "Decreases Risk": "#279477", "High Weight": "#4A6AD4"},
                            labels={"shap_value": "Contribution", "feature": ""})
            chart(figure, height=200, key="shap_drivers")
            st.dataframe(data.rename(columns={"feature": "Feature", "shap_value": "Contribution", "impact": "Direction", "feature_value": "Input value"}),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("No SHAP contributors were returned for this prediction.")
        st.caption("V1–V28 are anonymized PCA components. A negative contribution does not guarantee a safe transaction.")


def analysis(api_url, health, presets, select_preset, clear_history_cache):
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            title("Tested transaction profiles", "Genuine held-out dataset samples · select to assess")
            buttons = st.columns(3)
            for column, key, label, note in zip(buttons, ("coffee", "electronics", "fraud"),
                    ("Coffee Shop", "Electronics Store", "High-Risk Attack"),
                    ("$6.42 · legitimate", "$150.66 · elevated risk", "$727.91 · confirmed fraud")):
                with column:
                    st.button(label, key=f"preset_{key}", on_click=select_preset, args=(key,), use_container_width=True)
                    st.caption(note)
        with st.form("transaction_form"):
            title("Transaction details", "All 30 inputs are sent unchanged to the prediction API")
            st.text_input("Transaction ID", key="input_transaction_id")
            amount, time = st.columns(2)
            with amount:
                st.number_input("Amount ($)", min_value=.01, step=1.0, format="%.2f", key="input_Amount")
            with time:
                st.number_input("Time (seconds elapsed)", step=60.0, format="%.1f", key="input_Time")
            with st.expander("Anonymized PCA features · V1–V28", expanded=False):
                st.caption("These are numeric components, not named banking attributes. Presets populate the complete vector.")
                columns = st.columns(4)
                for index in range(1, 29):
                    with columns[(index - 1) % 4]:
                        st.number_input(f"V{index}", step=.1, format="%.4f", key=f"input_V{index}")
            submitted = st.form_submit_button("Analyze transaction", type="primary")
        st.caption("Preset provenance: Coffee index 173117 · Electronics 201083 · Fraud 211895. All belong to the held-out test split.")

    auto_trigger = st.session_state.pop("auto_trigger", False)
    should_run = submitted or auto_trigger
    if should_run:
        payload = {"transaction_id": st.session_state["input_transaction_id"]}
        payload.update({feature: float(st.session_state[f"input_{feature}"]) for feature in st.session_state["txn_features"]})
        st.session_state["txn_features"] = {key: value for key, value in payload.items() if key != "transaction_id"}
        st.session_state["txn_id"] = payload["transaction_id"]
        st.session_state["last_payload"] = payload
        st.session_state["prediction_error"] = None
        st.session_state["last_prediction"] = None
        with right, st.spinner("Assessing transaction…"):
            try:
                response = requests.post(f"{api_url}/predict", json=payload, timeout=5)
                if response.status_code == 200:
                    st.session_state["last_prediction"] = response.json()
                    st.session_state["prediction_threshold"] = health.get("optimal_threshold")
                    clear_history_cache()
                else:
                    st.session_state["prediction_error"] = f"Prediction API returned {response.status_code}: {response.text}"
            except requests.RequestException as exc:
                st.session_state["prediction_error"] = f"Could not reach the prediction API: {exc}"
    with right:
        if st.session_state.get("prediction_error"):
            st.error(st.session_state["prediction_error"])
        render_result(health)
    if st.session_state.get("last_payload"):
        with st.expander("Request inspection · exact last payload"):
            st.caption(f"POST {api_url}/predict · current result applies to this submitted vector")
            st.json(st.session_state["last_payload"])


def monitoring(frame, available, api_url, clear_history_cache):
    history_kpis(frame, available)
    st.write("")
    with st.container(border=True):
        title("Audit explorer", "Search and filter the most recent 500 records; this is not the all-time database total")
        search, tiers, flagged = st.columns([1.7, 1.3, 1])
        with search:
            query = st.text_input("Search transaction ID", placeholder="Search transaction ID…", key="audit_search")
        with tiers:
            selected = st.multiselect("Risk levels", list(RISK_COLORS), key="audit_tiers")
        with flagged:
            only_flagged = st.checkbox("Flagged only", key="audit_flagged")
        filtered = filter_history(frame, query, selected, only_flagged)
        st.caption(f"{len(filtered):,} of {len(frame):,} loaded records · timestamps in UTC")
        audit_table(filtered)
    first, second = st.columns(2)
    with first, st.container(border=True):
        title("Transactions by risk level", "Current filters applied")
        distribution(filtered, key="monitor_distribution")
    with second, st.container(border=True):
        title("Amount vs. risk score", "Current filters applied")
        if filtered.empty:
            empty("No matching records", "Chart data follows the filters above.")
        else:
            figure = px.scatter(filtered, x="amount", y="risk_score", color="risk_level", color_discrete_map=RISK_COLORS,
                                hover_data=["transaction_id"], labels={"amount": "Amount ($)", "risk_score": "Risk score", "risk_level": ""})
            figure.update_traces(marker=dict(size=7, opacity=.8))
            chart(figure, key="monitor_scatter")
    with st.container(border=True):
        title("Recorded risk activity", "Current filters applied · UTC")
        activity(filtered, key="monitor_activity")
    with st.expander("Demo audit history management"):
        st.caption("Historical hybrid-preset records preserve the inputs submitted at that time. The previous widget bug has been fixed; existing records are not rewritten.")
        st.warning("The reset below permanently deletes ALL records, including rows outside the current filters and 500-record window.")
        with st.form("clear_history_form"):
            confirmed = st.checkbox("I understand — permanently delete all audit records", key="confirm_clear")
            clear = st.form_submit_button("Clear Demo Audit History")
        if clear:
            if not confirmed:
                st.info("Confirm the checkbox before resetting demo history.")
            else:
                try:
                    response = requests.delete(f"{api_url}/history", timeout=5)
                    if response.status_code == 200:
                        clear_history_cache()
                        st.session_state["audit_notice"] = f"Deleted {response.json()['deleted_count']} audit records."
                        st.rerun()
                    else:
                        st.error(f"Reset failed: HTTP {response.status_code}")
                except requests.RequestException as exc:
                    st.error(f"Could not reach the audit API: {exc}")


def performance(health, metrics, frame):
    if not metrics:
        st.info("Model metrics are unavailable. Check that the API loaded its existing metric artifact.")
        return
    active, scope = st.columns([1, 1.7])
    with active, st.container(border=True):
        model_health(health, metrics, frame)
    with scope, st.container(border=True):
        title("Evaluation protocol", "Validation selects the model; test data estimates generalization")
        detail("Split", "70% train / 15% validation / 15% test")
        detail("Preparation", "Deduplicate before splitting; scaler fit on train only")
        detail("Model selection", "Validation average precision / PR-AUC")
        detail("Threshold objective", "Maximum validation F1 on a fixed candidate grid")
        st.caption("The selected threshold is not globally optimal and does not directly optimize monetary loss. Probabilities have not undergone a separate calibration step.")
    benchmark = metrics.get("validation_benchmark", {})
    if benchmark:
        with st.container(border=True):
            count = next(iter(benchmark.values())).get("total_evaluated", "—")
            title("Validation-set benchmark", f"{count:,} transactions · default threshold 0.50" if isinstance(count, int) else "Validation partition · default threshold 0.50")
            data = pd.DataFrame([{"Model": name, **row} for name, row in benchmark.items()])
            fields = ["Model", "pr_auc", "roc_auc", "precision", "recall", "f1_score", "false_positives", "false_negatives"]
            shown = data[fields].rename(columns={"pr_auc": "PR-AUC / AP", "roc_auc": "ROC-AUC", "precision": "Precision", "recall": "Recall", "f1_score": "F1", "false_positives": "FP", "false_negatives": "FN"})
            st.dataframe(shown, hide_index=True, use_container_width=True)
            figure = px.bar(data, x="Model", y=["pr_auc", "roc_auc", "f1_score", "recall"], barmode="group",
                            color_discrete_sequence=["#2855C5", "#82A6E2", "#AFBCD5", "#29476E"], labels={"value": "Score", "variable": "Metric"})
            chart(figure, height=240, key="model_benchmark")
    default = metrics.get("test_metrics_default_threshold", {})
    tuned = metrics.get("test_metrics_optimal_threshold", {})
    if default and tuned:
        with st.container(border=True):
            title("Held-out test · threshold comparison", f"{tuned.get('total_evaluated', '—')} transactions · independent of model / threshold selection")
            rows = []
            for label, key in [("Threshold", "threshold"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1_score"), ("True positives", "true_positives"), ("False positives", "false_positives"), ("False negatives", "false_negatives")]:
                rows.append({"Metric": label, "Default": default[key], "Validation-tuned": tuned[key]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            extra_tp = tuned["true_positives"] - default["true_positives"]
            extra_fp = tuned["false_positives"] - default["false_positives"]
            st.caption(f"Compared with the default on this test split: true positives change by {extra_tp:+d}, false positives by {extra_fp:+d}; F1 moves from {default['f1_score']:.4f} to {tuned['f1_score']:.4f}.")
    curve = metrics.get("threshold_curve_sample", [])
    if curve:
        with st.container(border=True):
            title("Validation threshold sweep", "Sampled candidate points from the saved artifact; not a new optimization run")
            data = pd.DataFrame(curve)
            figure = px.line(data, x="threshold", y=["precision", "recall", "f_beta"], color_discrete_sequence=["#2855C5", "#82A6E2", "#29476E"], labels={"value": "Score", "variable": "Metric"})
            if metrics.get("optimal_threshold") is not None:
                figure.add_vline(x=metrics["optimal_threshold"], line_dash="dot", line_color="#8490A2")
            chart(figure, key="threshold_curve")


def architecture(api_url):
    steps = [
        ("01 →", "Transaction input", "Streamlit submits Time, Amount and 28 anonymized PCA features to POST /predict."),
        ("02 →", "Validate & transform", "FastAPI / Pydantic validate the input. The saved preprocessor applies the fitted training transformations."),
        ("03 →", "Model inference", "The deployed model artifact estimates the fraud probability. No training runs during a prediction."),
        ("04 →", "Score & explain", "The risk engine applies the threshold and business tiers. SHAP computes feature contributions when initialized."),
        ("05 →", "Persist audit event", "The API commits the prediction and input feature JSON to SQLite before returning a successful response."),
        ("06", "Review & monitor", "The dashboard displays the response and retrieves persisted history and saved evaluation metrics."),
    ]
    st.markdown('<div class="flow">' + "".join(f'<div class="flow-step"><div class="flow-number">{html(n)}</div><h3>{html(name)}</h3><p>{html(desc)}</p></div>' for n, name, desc in steps) + '</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left, st.container(border=True):
        title("Training path", "Separate from live inference")
        st.write("Dataset → deduplication → stratified split → training-only scaling → candidate models → validation selection → held-out test evaluation → saved artifacts")
        detail("Candidates", "Logistic Regression · Random Forest · XGBoost")
        detail("Artifacts", "Model · preprocessor · metrics JSON")
    with right, st.container(border=True):
        title("Integration surface", "Existing API contracts are preserved")
        for method, route in [("GET", "/health"), ("POST", "/predict"), ("POST", "/batch-predict"), ("GET", "/history"), ("GET", "/metrics"), ("DELETE", "/history")]:
            detail(method, route)
        st.link_button("Open API documentation ↗", f"{api_url}/docs")


def system(health, health_ms, metrics, history_available, api_url):
    left, right = st.columns(2)
    with left, st.container(border=True):
        title("Service information", "Observed values from the current API")
        detail("API endpoint", api_url)
        detail("API status", health.get("status", "Unavailable"))
        detail("Model loaded", "Yes" if health.get("model_loaded") else "Not confirmed")
        detail("Model family", health.get("model_family", "—"))
        detail("Model version", health.get("version", "—"))
        detail("Decision threshold", number(health.get("optimal_threshold")))
        detail("Health-request round trip", f"{health_ms:.1f} ms" if health_ms is not None else "Unavailable")
        detail("Audit query", "Responding" if history_available else "Unavailable")
        detail("SHAP initialization status", "Not exposed by the health endpoint")
        st.caption("Health-request timing is not inference latency. SHAP contributors, when available, appear alongside an actual prediction.")
    with right, st.container(border=True):
        title("Data & model boundaries", "Local demonstration environment")
        data = metrics.get("data_summary", {})
        detail("Source dataset rows", f"{data['total_transactions']:,}" if "total_transactions" in data else "—")
        detail("Duplicate rows removed", data.get("duplicate_rows", "—"))
        detail("Dataset fraud rate", f"{data['fraud_percentage']:.4f}%" if "fraud_percentage" in data else "—")
        detail("Audit display window", "Most recent 500 records")
        st.caption("Displayed audit rates describe local/demo predictions, not the source dataset. V1–V28 carry no disclosed banking-field semantics.")
        st.warning("Local demo only: API access and history reset are unauthenticated. Do not expose this service publicly without appropriate security controls.")
    with st.expander("Example API request"):
        sample = {"transaction_id": "TXN-API-EXAMPLE", "Amount": 150.0, "Time": 4500.0, "V4": 1.2, "V14": -2.1}
        st.code(f"curl -X POST '{api_url}/predict' -H 'Content-Type: application/json' -d '{json.dumps(sample)}'", language="bash")
        st.caption("The API allows omitted PCA inputs to use their schema defaults. Genuine presets in Transaction Analysis always send all 30 features.")
