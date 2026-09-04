"""Presentation-only components; all model outputs come from the existing API."""
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RISK_COLORS = {
    "Low Risk": "#279477", "Medium Risk": "#C49636",
    "High Risk": "#E68A50", "Critical Risk": "#D75A68",
}
RISK_BACKGROUNDS = {
    "Low Risk": "#EBF7F1", "Medium Risk": "#FCF5E5",
    "High Risk": "#FFF0E7", "Critical Risk": "#FDEEF0",
}


def html(value):
    return escape(str(value), quote=True)


def title(label, caption=""):
    st.markdown(f'<div class="section-title">{html(label)}</div>'
                f'<div class="section-caption">{html(caption)}</div>', unsafe_allow_html=True)


def empty(heading, message):
    st.markdown(f'<div class="empty"><strong>{html(heading)}</strong><br>{html(message)}</div>', unsafe_allow_html=True)


def detail(label, value):
    st.markdown(f'<div class="detail-row"><span>{html(label)}</span>'
                f'<span class="detail-value">{html(value)}</span></div>', unsafe_allow_html=True)


def kpi(label, value, caption, rail=False):
    st.markdown(f'<div class="kpi{" kpi-rail" if rail else ""}"><div class="kpi-label">{html(label)}</div>'
                f'<div class="kpi-value">{html(value)}</div>'
                f'<div class="kpi-foot"><span class="kpi-marker"></span>{html(caption)}</div></div>', unsafe_allow_html=True)


def history_frame(records):
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame


def filter_history(frame, search="", tiers=None, flagged_only=False):
    result = frame.copy()
    if result.empty:
        return result
    if search.strip():
        result = result[result["transaction_id"].astype(str).str.contains(search.strip(), case=False, regex=False)]
    if tiers:
        result = result[result["risk_level"].isin(tiers)]
    if flagged_only:
        result = result[result["is_fraud"].eq(True)]
    return result


def history_kpis(frame, available=True, vertical=False):
    total = len(frame)
    flagged = int(frame["is_fraud"].sum()) if total else 0
    average = frame["risk_score"].mean() if total else 0
    alerts = int((frame["risk_score"] >= 70).sum()) if total else 0
    values = [
        ("Transactions audited", f"{total:,}", "Latest 500 records maximum"),
        ("Flagged transactions", f"{flagged:,}", f"{flagged / total * 100:.1f}% flagged rate" if total else "No predictions recorded"),
        ("Average risk score", f"{average:.2f}", "Model score · out of 100"),
        ("High / critical alerts", f"{alerts:,}", "Risk score of 70 or higher"),
    ]
    columns = [st.container() for _ in values] if vertical else st.columns(4)
    for column, (label, value, caption) in zip(columns, values):
        with column:
            kpi(label, value if available else "—", caption if available else "Audit service unavailable", rail=vertical)


def chart(figure, height=245, key=None, legend_top=False):
    figure.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Arial, sans-serif", size=11, color="#69788B"),
        legend=dict(orientation="h", y=-.15, x=0, font=dict(size=10)),
    )
    figure.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    figure.update_yaxes(gridcolor="#F0F2F7", zeroline=False, automargin=True)
    if legend_top:
        figure.update_layout(legend=dict(y=1.08, yanchor="bottom", title_text=""), margin=dict(t=35, b=12))
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False}, key=key)


def distribution(frame, key="distribution", height=245):
    if frame.empty:
        empty("No risk distribution yet", "Analyze a transaction to populate this view.")
        return
    counts = frame["risk_level"].value_counts().reindex(RISK_COLORS, fill_value=0)
    figure = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=.76, sort=False,
        marker=dict(colors=list(RISK_COLORS.values()), line=dict(color="white", width=3)),
        textinfo="none", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    figure.add_annotation(x=.5, y=.5, text=f'<b>{len(frame):,}</b><br><span style="font-size:10px">events</span>', showarrow=False,
                          font=dict(size=17, color="#263650"))
    chart(figure, height=height, key=key)


def activity(frame, key="activity", height=245):
    if frame.empty or frame["timestamp"].notna().sum() == 0:
        empty("Your activity will appear here", "Risk scores are plotted from timestamped audit records.")
        return
    ordered = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    figure = px.line(ordered, x="timestamp", y="risk_score", markers=True,
                     hover_data=["transaction_id"], labels={"timestamp": "Time (UTC)", "risk_score": "Risk score"})
    figure.update_traces(line=dict(color="#2E3D50", width=2), marker=dict(size=5, color="#F07B25"), fill="tozeroy", fillcolor="rgba(46,61,80,.035)")
    figure.update_yaxes(range=[0, 105], dtick=25)
    chart(figure, height=height, key=key)


def recent_amount_events(frame, limit=12):
    """Keep each audited request distinct, including repeated transaction IDs."""
    if frame.empty:
        return frame.copy()
    events = frame.sort_values("id").tail(limit).copy()
    events["Audit event"] = events["id"].astype(str)
    events["Prediction"] = events["is_fraud"].map({True: "Flagged", False: "Not flagged"})
    return events


def assessed_amounts(frame, height=225):
    data = recent_amount_events(frame)
    if data.empty:
        empty("No assessed amounts yet", "Amounts come from recorded prediction requests, not settled payments.")
        return
    figure = px.bar(
        data, x="Audit event", y="amount", color="Prediction",
        color_discrete_map={"Not flagged": "#2E3D50", "Flagged": "#F07B25"},
        category_orders={"Audit event": data["Audit event"].tolist(), "Prediction": ["Not flagged", "Flagged"]},
        hover_data=["transaction_id", "risk_score"],
        labels={"amount": "Input amount ($)", "Audit event": "Audit event · oldest → newest"},
    )
    figure.update_layout(bargap=.4)
    figure.update_xaxes(type="category")
    chart(figure, height=height, key="overview_amounts", legend_top=True)


def audit_table(frame, limit=None):
    if frame.empty:
        empty("No matching transactions", "Try another filter or analyze a transaction.")
        return
    columns = [c for c in ["transaction_id", "timestamp", "amount", "fraud_probability", "risk_score", "risk_level", "is_fraud", "decision", "latency_ms"] if c in frame]
    shown = frame[columns].head(limit).copy() if limit else frame[columns].copy()
    shown["fraud_probability"] = shown["fraud_probability"] * 100
    st.dataframe(
        shown, hide_index=True, use_container_width=True,
        column_config={
            "transaction_id": st.column_config.TextColumn("Transaction ID", width="medium"),
            "timestamp": st.column_config.DatetimeColumn("Timestamp · UTC", format="DD MMM HH:mm:ss"),
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "fraud_probability": st.column_config.NumberColumn("Probability", format="%.2f%%"),
            "risk_score": st.column_config.NumberColumn("Risk score", format="%.2f"),
            "risk_level": "Risk level", "is_fraud": "Flagged", "decision": "Recommendation",
            "latency_ms": st.column_config.NumberColumn("Latency", format="%.2f ms"),
        },
    )


def number(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (float, int)) else "—"


def model_health(health, metrics, frame):
    title("Model health", "Active inference service")
    st.markdown(f'<div class="model-title">{html(metrics.get("best_model_name", health.get("model_family", "Unavailable")))}</div>'
                f'<div class="model-version">Version {html(health.get("version", "—"))} · held-out test metrics below</div>', unsafe_allow_html=True)
    test = metrics.get("test_metrics_optimal_threshold", {})
    detail("PR-AUC / Average precision", number(test.get("pr_auc")))
    detail("ROC-AUC", number(test.get("roc_auc")))
    detail("Decision threshold", number(health.get("optimal_threshold")))
    latency = frame["latency_ms"].dropna().mean() if not frame.empty else None
    detail("Mean recorded inference", f"{latency:.2f} ms" if latency is not None and pd.notna(latency) else "No samples")
