"""
app.py
======
Streamlit frontend for Credit Card Fraud Detection.

Run with:
    streamlit run app.py

This module only handles the UI; the logic lives in:
    preprocess.py      loading and feature preparation
    prediction.py      scoring and classification
    evaluation.py      confusion matrix and metrics
    visualization.py   fraud statistics and probability chart
    download.py        CSV export
"""

from pathlib import Path

import streamlit as st

from download import render_download_button
from evaluation import confusion_matrix_frame
from prediction import load_artifacts, score_transactions
from preprocess import load_transactions
from visualization import (
    fraud_summary,
    highlight_fraud,
    probability_distribution,
)

MODEL_DIR = Path(__file__).parent / "models"
PIPELINE_PATH = MODEL_DIR / "best_pipeline.pkl"

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
)

# Compact header so results stay visible in short panes.
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    .block-container h1 { font-size: 1.6rem; margin: 0 0 0.25rem 0; padding: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💳 Fraud Detector")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        "1. Upload a CSV file containing transaction data.\n"
        "2. The app scores every transaction.\n"
        "3. Review flagged transactions highlighted in red.\n"
    )


# ─── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not PIPELINE_PATH.exists():
        return None, None
    return load_artifacts(MODEL_DIR)


pipeline, threshold = load_model()

# ─── Main UI ───────────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection")
st.caption("Upload a CSV file of transactions to screen for fraud.")

if pipeline is None:
    st.error(
        "⚠️ **Model not found.**\n\n"
        "Please train the model first with `python model_training.py`, or place "
        "`best_pipeline.pkl` and `threshold.json` in the `models/` folder.\n\n"
        "See `README.md` for step-by-step instructions."
    )
    st.stop()

st.success(f"✅ Model loaded — using decision threshold **{threshold:.4f}**")

# ─── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    help="Must contain V1–V28 and Amount columns.",
)
st.caption("Maximum file size: 700 MB.")

if uploaded_file is not None:
    try:
        df = load_transactions(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    with st.spinner("Scoring transactions…"):
        df_out, proba, preds = score_transactions(df, pipeline, threshold)

    summary = fraud_summary(preds)

    # ─── Confusion Matrix (if Class column exists) ────────────────────────────
    if "Class" in df.columns:
        st.markdown("### Confusion Matrix")
        st.table(confusion_matrix_frame(df["Class"].values, preds))

    # ─── Summary Cards ─────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{summary['total']:,}")
    col2.metric("Flagged as Fraud", f"{summary['fraud']:,}", delta=None)
    col3.metric("Fraud Rate", f"{summary['fraud_rate']:.3f}%")

    # ─── Fraud Probability Distribution ────────────────────────────────────────
    st.markdown("### Fraud Probability Distribution")
    st.bar_chart(probability_distribution(proba))

    # ─── Results Table ─────────────────────────────────────────────────────────
    st.markdown("### Transaction Results")

    show_only_fraud = st.checkbox("Show flagged transactions only", value=False)
    display_df = df_out[preds == 1] if show_only_fraud else df_out

    MAX_PREVIEW_ROWS = 5_000
    preview_df = display_df.head(MAX_PREVIEW_ROWS)
    if len(display_df) > MAX_PREVIEW_ROWS:
        st.caption(
            f"Showing the first {MAX_PREVIEW_ROWS:,} of {len(display_df):,} rows. "
            "Download the CSV below for the full results."
        )

    # Display columns: put key columns first (skip any that are absent).
    key_cols = [
        c for c in ["fraud_probability", "Prediction", "Amount", "Time"]
        if c in preview_df.columns
    ]
    other_cols = [c for c in preview_df.columns if c not in key_cols]
    table = preview_df[key_cols + other_cols]

    # Styling is slow and Pandas Styler refuses to render more than
    # 262,144 cells; skip it for large tables instead of crashing/hanging.
    if table.size <= 50_000:
        table = table.style.apply(highlight_fraud, axis=1).format(
            {"fraud_probability": "{:.4f}"}
        )

    st.dataframe(table, use_container_width=True)

    # ─── Download Results ──────────────────────────────────────────────────────
    render_download_button(df_out)
