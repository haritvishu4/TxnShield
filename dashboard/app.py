"""Local risk-operations dashboard; the backend and model contracts are unchanged."""

import os
import sys
import time
import uuid
from pathlib import Path

# Make the project root importable on Streamlit Community Cloud
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
# pyrefly: ignore [missing-import]
import streamlit as st

from dashboard import pages
from dashboard.components import history_frame, html
from dashboard.state import sync_preset_state


st.set_page_config(
    page_title="TxnShield | Transaction Fraud Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>"
    + Path(__file__).with_name("theme.css").read_text(encoding="utf-8")
    + "</style>",
    unsafe_allow_html=True,
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


# Genuine test dataset vectors for realistic presets (held-out test split)
DATASET_PRESETS = {
    "coffee": {
        "name": "☕ Coffee Shop (Legitimate - Test Index 173117)",
        "desc": "Genuine legitimate retail transaction ($6.42) from held-out test split (Class 0).",
        "features": {
            "Time": 121634.0,
            "Amount": 6.42,
            "V1": 2.1615,
            "V2": -0.0827,
            "V3": -2.4382,
            "V4": 0.2243,
            "V5": 0.7392,
            "V6": -0.9843,
            "V7": 0.5755,
            "V8": -0.3377,
            "V9": 0.8001,
            "V10": 0.0597,
            "V11": -2.1355,
            "V12": -1.2218,
            "V13": -2.3872,
            "V14": 1.0712,
            "V15": 0.1743,
            "V16": -0.5358,
            "V17": -0.2903,
            "V18": -0.0305,
            "V19": 0.4980,
            "V20": -0.3929,
            "V21": 0.0434,
            "V22": 0.2575,
            "V23": -0.1740,
            "V24": -1.0771,
            "V25": 0.6042,
            "V26": 0.0073,
            "V27": -0.0668,
            "V28": -0.0902,
        },
    },

    "electronics": {
        "name": "💻 Electronics Store (Elevated Risk - Test Index 201083)",
        "desc": "Genuine elevated risk transaction ($150.66) from test split with borderline feature pattern.",
        "features": {
            "Time": 134047.0,
            "Amount": 150.66,
            "V1": 1.7608,
            "V2": -0.1369,
            "V3": -2.4043,
            "V4": 0.8479,
            "V5": 0.5088,
            "V6": -0.9360,
            "V7": 0.4497,
            "V8": -0.2641,
            "V9": 0.8600,
            "V10": -1.1372,
            "V11": -0.4757,
            "V12": -0.0367,
            "V13": -0.3540,
            "V14": -2.0503,
            "V15": -0.0989,
            "V16": 0.1424,
            "V17": 1.6158,
            "V18": 0.5625,
            "V19": -0.0849,
            "V20": 0.1224,
            "V21": -0.1051,
            "V22": -0.3704,
            "V23": -0.0368,
            "V24": 0.4872,
            "V25": 0.1601,
            "V26": -0.3375,
            "V27": -0.0154,
            "V28": 0.0240,
        },
    },

    "fraud": {
        "name": "🚨 High-Risk Attack (Confirmed Fraud - Test Index 211895)",
        "desc": "Genuine fraudulent credit card transaction ($727.91) from held-out test split (Class 1).",
        "features": {
            "Time": 138942.0,
            "Amount": 727.91,
            "V1": -2.3563,
            "V2": 1.7464,
            "V3": -6.3746,
            "V4": 1.7722,
            "V5": -3.4393,
            "V6": 1.4578,
            "V7": -0.3626,
            "V8": 1.4438,
            "V9": -1.9274,
            "V10": -6.5647,
            "V11": 2.4508,
            "V12": -5.6941,
            "V13": -1.1555,
            "V14": -7.1322,
            "V15": -0.0596,
            "V16": -4.5966,
            "V17": -5.5221,
            "V18": -3.5291,
            "V19": -0.6634,
            "V20": 0.1948,
            "V21": 0.8579,
            "V22": 0.6212,
            "V23": 0.9648,
            "V24": -0.6194,
            "V25": -1.7326,
            "V26": 0.1084,
            "V27": 1.1308,
            "V28": 0.4157,
        },
    },
}


@st.cache_data(ttl=5, show_spinner=False)
def fetch_health(url):
    start = time.perf_counter()

    try:
        response = requests.get(
            f"{url}/health",
            timeout=2,
        )

        if response.status_code == 200:
            return response.json(), (time.perf_counter() - start) * 1000

    except (requests.RequestException, ValueError):
        pass

    return {}, None


@st.cache_data(ttl=5, show_spinner=False)
def fetch_history(url):
    try:
        response = requests.get(
            f"{url}/history?limit=500",
            timeout=3,
        )

        if response.status_code == 200:
            return response.json(), True

    except (requests.RequestException, ValueError):
        pass

    return [], False


@st.cache_data(ttl=10, show_spinner=False)
def fetch_metrics(url):
    try:
        response = requests.get(
            f"{url}/metrics",
            timeout=3,
        )

        if response.status_code == 200:
            return response.json()

    except (requests.RequestException, ValueError):
        pass

    return {}


def refresh_data():
    fetch_health.clear()
    fetch_history.clear()
    fetch_metrics.clear()


def select_preset(preset_key):
    sync_preset_state(
        st.session_state,
        preset_key,
        DATASET_PRESETS,
    )


# Keep canonical inputs separate from the conditional page widgets.
# Reassigning existing widget keys preserves submitted values across navigation.
st.session_state.setdefault(
    "txn_features",
    DATASET_PRESETS["coffee"]["features"].copy(),
)

st.session_state.setdefault(
    "active_preset_name",
    DATASET_PRESETS["coffee"]["name"],
)

st.session_state.setdefault(
    "txn_id",
    f"TXN-{uuid.uuid4().hex[:8].upper()}",
)

st.session_state.setdefault(
    "auto_trigger",
    False,
)

for feature, value in st.session_state["txn_features"].items():
    key = f"input_{feature}"

    st.session_state[key] = st.session_state.get(
        key,
        value,
    )


st.session_state["input_transaction_id"] = st.session_state.get(
    "input_transaction_id",
    st.session_state["txn_id"],
)


PAGE_DESCRIPTIONS = {
    "Overview": "A clear view of transaction risk, model performance and recent activity.",
    "Transaction Analysis": "Investigate a transaction. Understand the score. Decide the next step.",
    "Monitoring & Audits": "Explore persisted predictions and focus on the events that need attention.",
    "Model Performance": "Transparent evaluation, from validation selection to held-out test results.",
    "Architecture": "From a transaction input to an explainable, auditable risk assessment.",
    "System Information": "Service status, dataset context and the boundaries of this local demo.",
}


with st.sidebar:
    st.markdown(
        '<div class="brand">'
        '<div class="brand-icon">TΞ</div>'
        '<div>'
        '<div class="brand-title">TxnShield</div>'
        '<div class="brand-sub">TRANSACTION FRAUD INTELLIGENCE</div>'
        '</div>'
        '</div>'
        '<div class="nav-caption">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        list(PAGE_DESCRIPTIONS),
        key="page",
        label_visibility="collapsed",
    )

    st.write("")

    st.button(
        "Refresh data",
        key="refresh_data",
        on_click=refresh_data,
        use_container_width=True,
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-note">'
        '<strong>Local demo workspace</strong><br>'
        'Predictions support review.<br>'
        'They do not authorize an account hold or confirm fraud.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption("ML-powered · Human-reviewed")


health, health_ms = fetch_health(
    API_BASE_URL,
)

records, history_available = fetch_history(
    API_BASE_URL,
)

metrics = fetch_metrics(
    API_BASE_URL,
)

frame = history_frame(
    records,
)

online = (
    bool(
        health.get(
            "model_loaded",
        )
    )
    and health.get(
        "status",
        "healthy",
    )
    == "healthy"
)

model_name = metrics.get(
    "best_model_name",
    health.get(
        "model_family",
        "Model unavailable",
    ),
)


st.markdown(
    f'<div class="page-header">'
    f'<div>'
    f'<div class="eyebrow">WORKSPACE / TRANSACTION INTELLIGENCE</div>'
    f'<h1>{html(page)}</h1>'
    f'<div class="page-description">{html(PAGE_DESCRIPTIONS[page])}</div>'
    f'</div>'
    f'<div class="header-status">'
    f'<div class="health-pill {"" if online else "health-offline"}">'
    f'<span class="health-dot"></span>'
    f'{"Model online" if online else "Service unavailable"}'
    f'</div>'
    f'<div class="header-model">{html(model_name)}</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)


if not online:
    st.warning(
        f"The API is offline or the model is not loaded. "
        f"Check the service at {API_BASE_URL}."
    )


if (
    not history_available
    and page in (
        "Overview",
        "Monitoring & Audits",
    )
):
    st.info(
        "Audit history could not be retrieved. "
        "Unavailable totals are shown as —, not zero."
    )


if st.session_state.get(
    "audit_notice",
):
    st.success(
        st.session_state.pop(
            "audit_notice",
        )
    )


if page == "Overview":
    pages.overview(
        health,
        metrics,
        frame,
        history_available,
    )

elif page == "Transaction Analysis":
    pages.analysis(
        API_BASE_URL,
        health,
        DATASET_PRESETS,
        select_preset,
        fetch_history.clear,
    )

elif page == "Monitoring & Audits":
    pages.monitoring(
        frame,
        history_available,
        API_BASE_URL,
        fetch_history.clear,
    )

elif page == "Model Performance":
    pages.performance(
        health,
        metrics,
        frame,
    )

elif page == "Architecture":
    pages.architecture(
        API_BASE_URL,
    )

elif page == "System Information":
    pages.system(
        health,
        health_ms,
        metrics,
        history_available,
        API_BASE_URL,
    )


st.markdown(
    '<div class="footnote">'
    'TXNSHIELD · Transaction Fraud Intelligence · '
    'Model outputs are advisory, not a guarantee.'
    '</div>',
    unsafe_allow_html=True,
)