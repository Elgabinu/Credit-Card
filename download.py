"""
download.py
===========
Downloading prediction results as CSV.

Usage (Streamlit):
    from download import predictions_csv, render_download_button

    render_download_button(df_out)

Usage (plain Python):
    data = predictions_csv(df_out)
    open("predictions.csv", "wb").write(data)
"""

import pandas as pd

DEFAULT_FILE_NAME = "fraud_predictions.csv"


def predictions_csv(df: pd.DataFrame, index: bool = False) -> bytes:
    """Serialise a predictions DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=index).encode("utf-8")


def render_download_button(
    df: pd.DataFrame,
    file_name: str = DEFAULT_FILE_NAME,
    label: str = "⬇️ Download predictions as CSV",
    mime: str = "text/csv",
):
    """
    Streamlit download button for the full (untruncated) results.

    Streamlit is imported lazily so this module can be used from scripts.
    """
    import streamlit as st

    return st.download_button(
        label=label,
        data=predictions_csv(df),
        file_name=file_name,
        mime=mime,
    )
