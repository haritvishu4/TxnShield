import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Fraud Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-low { background-color: #2e7d32; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-medium { background-color: #f9a825; color: black; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-high { background-color: #e65100; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-critical { background-color: #c62828; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Genuine test dataset vectors for realistic presets (held-out test split)
DATASET_PRESETS = {
    "coffee": {
        "name": "☕ Coffee Shop (Legitimate - Test Index 173117)",
        "desc": "Genuine legitimate retail transaction ($6.42) from held-out test split (Class 0).",
        "features": {
            "Time": 121634.0, "Amount": 6.42, "V1": 2.1615, "V2": -0.0827, "V3": -2.4382, "V4": 0.2243,
            "V5": 0.7392, "V6": -0.9843, "V7": 0.5755, "V8": -0.3377, "V9": 0.8001, "V10": 0.0597,
            "V11": -2.1355, "V12": -1.2218, "V13": -2.3872, "V14": 1.0712, "V15": 0.1743, "V16": -0.5358,
            "V17": -0.2903, "V18": -0.0305, "V19": 0.4980, "V20": -0.3929, "V21": 0.0434, "V22": 0.2575,
            "V23": -0.1740, "V24": -1.0771, "V25": 0.6042, "V26": 0.0073, "V27": -0.0668, "V28": -0.0902
        }
    },
    "electronics": {
        "name": "💻 Electronics Store (Elevated Risk - Test Index 201083)",
        "desc": "Genuine elevated risk transaction ($150.66) from test split with borderline feature pattern.",
        "features": {
            "Time": 134047.0, "Amount": 150.66, "V1": 1.7608, "V2": -0.1369, "V3": -2.4043, "V4": 0.8479,
            "V5": 0.5088, "V6": -0.9360, "V7": 0.4497, "V8": -0.2641, "V9": 0.8600, "V10": -1.1372,
            "V11": -0.4757, "V12": -0.0367, "V13": -0.3540, "V14": -2.0503, "V15": -0.0989, "V16": 0.1424,
            "V17": 1.6158, "V18": 0.5625, "V19": -0.0849, "V20": 0.1224, "V21": -0.1051, "V22": -0.3704,
            "V23": -0.0368, "V24": 0.4872, "V25": 0.1601, "V26": -0.3375, "V27": -0.0154, "V28": 0.0240
        }
    },
    "fraud": {
        "name": "🚨 High-Risk Attack (Confirmed Fraud - Test Index 211895)",
        "desc": "Genuine fraudulent credit card transaction ($727.91) from held-out test split (Class 1).",
        "features": {
            "Time": 138942.0, "Amount": 727.91, "V1": -2.3563, "V2": 1.7464, "V3": -6.3746, "V4": 1.7722,
            "V5": -3.4393, "V6": 1.4578, "V7": -0.3626, "V8": 1.4438, "V9": -1.9274, "V10": -6.5647,
            "V11": 2.4508, "V12": -5.6941, "V13": -1.1555, "V14": -7.1322, "V15": -0.0596, "V16": -4.5966,
            "V17": -5.5221, "V18": -3.5291, "V19": -0.6634, "V20": 0.1948, "V21": 0.8579, "V22": 0.6212,
            "V23": 0.9648, "V24": -0.6194, "V25": -1.7326, "V26": 0.1084, "V27": 1.1308, "V28": 0.4157
        }
    }
}

# Initialize session state for all 30 model features
if "txn_features" not in st.session_state:
    st.session_state["txn_features"] = DATASET_PRESETS["coffee"]["features"].copy()
    st.session_state["active_preset_name"] = DATASET_PRESETS["coffee"]["name"]
    st.session_state["txn_id"] = f"TXN-{np.random.randint(10000, 99999)}"
    st.session_state["auto_trigger"] = True

def select_preset(preset_key):
    preset = DATASET_PRESETS[preset_key]
    st.session_state["txn_features"] = preset["features"].copy()
    st.session_state["active_preset_name"] = preset["name"]
    st.session_state["txn_id"] = f"TXN-PRESET-{preset_key.upper()}"
    st.session_state["auto_trigger"] = True

# Helper functions to check API health and fetch logs
@st.cache_data(ttl=5)
def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=5)
def fetch_history():
    try:
        resp = requests.get(f"{API_BASE_URL}/history?limit=100", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []

@st.cache_data(ttl=10)
def fetch_metrics():
    try:
        resp = requests.get(f"{API_BASE_URL}/metrics", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}

health = check_api_health()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Risk Intelligence")
    st.markdown("**Real-Time Financial Fraud Detection**")
    st.divider()

    if health and health.get("model_loaded"):
        st.success(f"🟢 **API Online** ({health.get('model_family')})")
        st.info(f"Decision Threshold: **τ\* = {health.get('optimal_threshold', 0.5)}**")
    else:
        st.error("🔴 **API Offline / Loading**")
        st.caption("Ensure FastAPI is running on `http://127.0.0.1:8000`")

    st.markdown("---")
    st.markdown("### Risk Tier Definitions")
    st.markdown("🟢 **0 - 29**: Low Risk (Auto-Approve)")
    st.markdown("🟡 **30 - 69**: Medium Risk (2FA Step-up)")
    st.markdown("🟠 **70 - 89**: High Risk (Analyst Review)")
    st.markdown("🔴 **90 - 100**: Critical Risk (Freeze)")

# Main Navigation Tabs
tab_sim, tab_monitor, tab_benchmark, tab_arch = st.tabs([
    "🎯 Real-Time Transaction Simulator",
    "📊 Transaction Monitoring & Audits",
    "📈 Model Evaluation & Benchmarks",
    "🏗️ Architecture & Integration"
])

# -------------------------------------------------------------
# TAB 1: TRANSACTION SIMULATOR
# -------------------------------------------------------------
with tab_sim:
    st.subheader("Simulate & Score Financial Transaction")
    st.write("Select a genuine dataset preset or manually adjust features to test real-time risk scoring.")

    # Preset transaction buttons with explicit callbacks
    st.markdown("##### ⚡ Quick-Load Tested Dataset Presets:")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.button("☕ Preset: Coffee Shop ($6.42)", on_click=select_preset, args=("coffee",), use_container_width=True)
        st.caption("Genuine Legitimate (Test Split Index 173117)")
    with col_p2:
        st.button("💻 Preset: Electronics Store ($150.66)", on_click=select_preset, args=("electronics",), use_container_width=True)
        st.caption("Genuine Elevated Risk (Test Split Index 201083)")
    with col_p3:
        st.button("🚨 Preset: High-Risk Attack ($727.91)", on_click=select_preset, args=("fraud",), use_container_width=True)
        st.caption("Genuine Confirmed Fraud (Test Split Index 211895)")

    st.info(f"Active Selected Profile: **{st.session_state.get('active_preset_name', 'Custom')}**")

    # Transaction Input Form
    with st.form(key="transaction_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        current_feat = st.session_state["txn_features"]

        with col_f1:
            txn_id = st.text_input("Transaction ID", value=st.session_state.get("txn_id", "TXN-DEMO-001"))
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=float(current_feat["Amount"]), step=1.0, format="%.2f")
        with col_f2:
            time_sec = st.number_input("Time (Seconds elapsed)", value=float(current_feat["Time"]), step=60.0, format="%.1f")
            v4 = st.number_input("V4 (Anonymized PCA Feature)", value=float(current_feat["V4"]), step=0.1, format="%.4f")
        with col_f3:
            v10 = st.number_input("V10 (Anonymized PCA Feature)", value=float(current_feat["V10"]), step=0.1, format="%.4f")
            v14 = st.number_input("V14 (Anonymized PCA Feature)", value=float(current_feat["V14"]), step=0.1, format="%.4f")

        with st.expander("Complete Model Feature Vector (V1 - V28 Anonymized PCA Components)", expanded=False):
            st.caption("All 28 anonymized PCA features are populated with genuine values from the dataset.")
            exp_cols = st.columns(4)
            full_v_inputs = {}
            for i in range(1, 29):
                key = f"V{i}"
                if key in ["V4", "V10", "V14"]:
                    continue
                col_idx = (i - 1) % 4
                with exp_cols[col_idx]:
                    val = float(current_feat.get(key, 0.0))
                    full_v_inputs[key] = st.number_input(
                        f"{key} (Anonymized PCA Feature)",
                        value=val,
                        step=0.1,
                        format="%.4f",
                        key=f"input_{key}"
                    )

        manual_submit = st.form_submit_button("🛡️ Run Fraud Detection & Risk Scoring", type="primary", use_container_width=True)

    # Determine if execution is triggered (either manual submit or preset button click)
    should_run = manual_submit or st.session_state.get("auto_trigger", False)
    if st.session_state.get("auto_trigger", False):
        st.session_state["auto_trigger"] = False

    if should_run:
        # Construct complete 30-feature model payload
        payload = {
            "transaction_id": txn_id,
            "Time": float(time_sec),
            "Amount": float(amount),
            "V4": float(v4),
            "V10": float(v10),
            "V14": float(v14),
            **full_v_inputs
        }
        # Update session state to keep in sync
        for k in current_feat:
            if k in payload:
                st.session_state["txn_features"][k] = payload[k]

        with st.spinner("Analyzing risk via trained machine learning ensemble..."):
            try:
                resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
                
                # --- TEMPORARY DEBUG OUTPUT SECTION ---
                with st.expander("🛠️ Debug: Exact Payload Sent to FastAPI (/predict)", expanded=True):
                    st.write(f"**Endpoint:** `POST {API_BASE_URL}/predict`")
                    st.write(f"**Preset Source:** `{st.session_state.get('active_preset_name', 'Custom')}`")
                    st.write(f"**HTTP Response Code:** `{resp.status_code}`")
                    st.markdown("**Complete 30-Feature JSON Payload:**")
                    st.json(payload)
                # --------------------------------------

                if resp.status_code == 200:
                    res = resp.json()
                    st.success("Analysis Complete!")

                    # Result Summary Cards
                    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
                    with r_col1:
                        st.metric("Fraud Probability", f"{res['fraud_probability']*100:.2f}%")
                    with r_col2:
                        st.metric("Risk Score", f"{res['risk_score']:.2f} / 100")
                    with r_col3:
                        badge_class = f"badge-{res['risk_level'].split()[0].lower()}"
                        st.markdown(f"**Risk Level:** <span class='{badge_class}'>{res['risk_level']}</span>", unsafe_allow_html=True)
                    with r_col4:
                        st.metric("Inference Latency", f"{res['latency_ms']} ms")

                    st.markdown(f"### Recommended Action: **`{res['decision']}`**")

                    # Score Gauge Visualization
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=res["risk_score"],
                        title={"text": f"Risk Score Indicator: {res['risk_level']}"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#1E88E5"},
                            "steps": [
                                {"range": [0, 30], "color": "#c8e6c9"},
                                {"range": [30, 70], "color": "#fff59d"},
                                {"range": [70, 90], "color": "#ffe0b2"},
                                {"range": [90, 100], "color": "#ffcdd2"}
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 70
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Explainability: SHAP Contributions
                    st.subheader("🔍 Prediction Explainability (SHAP Risk Factors)")
                    st.write("Feature attributions quantifying marginal impact on the prediction (positive = elevates risk, negative = assures safety):")

                    if res.get("top_risk_drivers"):
                        df_shap = pd.DataFrame(res["top_risk_drivers"])
                        fig_bar = px.bar(
                            df_shap,
                            x="shap_value",
                            y="feature",
                            orientation="h",
                            color="impact",
                            color_discrete_map={"Increases Risk": "#e53935", "Decreases Risk": "#43a047", "High Weight": "#1E88E5"},
                            text="feature_value",
                            title="Top SHAP Feature Contributions"
                        )
                        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300)
                        st.plotly_chart(fig_bar, use_container_width=True)

                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Failed to connect to API backend: {e}. Ensure FastAPI is running on {API_BASE_URL}")

# -------------------------------------------------------------
# TAB 2: MONITORING & AUDITS
# -------------------------------------------------------------
with tab_monitor:
    st.subheader("Transaction Monitoring & Audit Trail")
    history_data = fetch_history()

    if history_data:
        df_hist = pd.DataFrame(history_data)

        # Overview metric cards
        m1, m2, m3, m4 = st.columns(4)
        total_txns = len(df_hist)
        fraud_txns = (df_hist["is_fraud"] == True).sum()
        avg_score = df_hist["risk_score"].mean()
        high_risk_txns = (df_hist["risk_score"] >= 70).sum()

        with m1:
            st.metric("Total Audited", f"{total_txns:,}")
        with m2:
            st.metric("Flagged Fraud", f"{fraud_txns:,}", f"{(fraud_txns/total_txns)*100:.1f}%")
        with m3:
            st.metric("Average Risk Score", f"{avg_score:.1f} / 100")
        with m4:
            st.metric("High/Critical Alerts", f"{high_risk_txns:,}")

        # Distribution Chart
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_pie = px.pie(
                df_hist,
                names="risk_level",
                title="Transactions by Risk Level",
                color="risk_level",
                color_discrete_map={
                    "Low Risk": "#43a047",
                    "Medium Risk": "#fbc02d",
                    "High Risk": "#fb8c00",
                    "Critical Risk": "#e53935"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_c2:
            fig_scatter = px.scatter(
                df_hist,
                x="amount",
                y="risk_score",
                color="risk_level",
                hover_data=["transaction_id", "decision"],
                title="Transaction Amount vs Risk Score",
                color_discrete_map={
                    "Low Risk": "#43a047",
                    "Medium Risk": "#fbc02d",
                    "High Risk": "#fb8c00",
                    "Critical Risk": "#e53935"
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Interactive Table
        st.markdown("### Recent Transaction Logs")
        st.dataframe(
            df_hist[["transaction_id", "amount", "fraud_probability", "risk_score", "risk_level", "decision", "latency_ms"]],
            use_container_width=True
        )
    else:
        st.info("No transaction history recorded yet. Run the simulation or seed the database using `python scripts/seed_db.py`.")

# -------------------------------------------------------------
# TAB 3: MODEL BENCHMARK & EVALUATION
# -------------------------------------------------------------
with tab_benchmark:
    st.subheader("Model Evaluation & Imbalanced Benchmark")
    metrics_summary = fetch_metrics()

    if metrics_summary and "validation_benchmark" in metrics_summary:
        val_bm = metrics_summary["validation_benchmark"]
        df_bench = pd.DataFrame([
            {"Model": k, **v} for k, v in val_bm.items()
        ])

        st.markdown(f"**Selected Best Production Model:** `{metrics_summary.get('best_model_name')}`")
        st.markdown(f"**Optimal Decision Threshold:** `τ* = {metrics_summary.get('optimal_threshold')}` (Optimized on Validation Set)")

        # Benchmark Comparison Table
        st.dataframe(
            df_bench[["Model", "pr_auc", "roc_auc", "precision", "recall", "f1_score", "true_positives", "false_positives", "false_negatives"]],
            use_container_width=True
        )

        # Comparison Chart
        fig_comp = px.bar(
            df_bench,
            x="Model",
            y=["pr_auc", "roc_auc", "f1_score", "recall"],
            barmode="group",
            title="Model Metric Benchmark (Handling Severe Imbalance)"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Test Set Performance comparison
        st.markdown("### Test Set Generalization (Default vs Optimal Threshold)")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            def_m = metrics_summary.get("test_metrics_default_threshold", {})
            st.markdown("#### Default Threshold (τ = 0.50)")
            st.write(f"- **F1 Score:** {def_m.get('f1_score', 'N/A')}")
            st.write(f"- **Recall:** {def_m.get('recall', 'N/A')}")
            st.write(f"- **Precision:** {def_m.get('precision', 'N/A')}")
            st.write(f"- **False Negatives (Missed Fraud):** {def_m.get('false_negatives', 'N/A')}")

        with t_col2:
            opt_m = metrics_summary.get("test_metrics_optimal_threshold", {})
            st.markdown(f"#### Optimal Threshold (τ* = {metrics_summary.get('optimal_threshold')})")
            st.write(f"- **F1 Score:** {opt_m.get('f1_score', 'N/A')}")
            st.write(f"- **Recall:** {opt_m.get('recall', 'N/A')}")
            st.write(f"- **Precision:** {opt_m.get('precision', 'N/A')}")
            st.write(f"- **False Negatives (Missed Fraud):** {opt_m.get('false_negatives', 'N/A')}")

    else:
        st.info("Model metrics not loaded yet. Run `python scripts/run_pipeline.py` to train and evaluate candidate models.")

# -------------------------------------------------------------
# TAB 4: SYSTEM ARCHITECTURE
# -------------------------------------------------------------
with tab_arch:
    st.subheader("System Architecture & API Integration")
    st.markdown("""
    The system decouples data ingestion, preprocessing, modeling, inference, risk scoring, and audit logging into a modular architecture:
    
    1. **Preprocessing Pipeline**: Prevents data leakage with stratified 70/15/15 split and robust scaling on transaction volume and temporal features.
    2. **Class Imbalance Strategy**: Employs dynamic class-weighting and validation-based decision threshold optimization ($\tau^*$).
    3. **Explainable AI (SHAP)**: Translates black-box ML outputs into human-auditable risk drivers.
    4. **FastAPI & SQLite**: Exposes sub-20ms prediction endpoint and persists transactions for regulatory compliance.
    """)

    st.markdown("### Sample API Request (`curl`)")
    st.code("""
curl -X POST "http://127.0.0.1:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "transaction_id": "TXN-API-990",
       "Amount": 1250.00,
       "Time": 4500.0,
       "V4": 3.8,
       "V10": -2.9,
       "V14": -4.1
     }'
    """, language="bash")
