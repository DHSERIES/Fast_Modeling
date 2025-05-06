# eda.py
import streamlit as st
import pandas as pd
import sweetviz as sv
import os

REPORT_DIR = "report/"

@st.cache_data
def make_report(df, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    report = sv.analyze(df)
    out = os.path.join(report_dir, "sweetviz_report.html")
    report.show_html(out)
    return out

def show_eda():
    st.title("📊 EDA with Sweetviz")

    path = st.session_state.get("dataset_path", None)
    if not path:
        st.warning("Please upload or select a dataset on the Home / Import Data page first.")
        return

    df = pd.read_csv(path)
    st.write("Preview:", df.head())

    if st.button("Generate Sweetviz report"):
        report_path = make_report(df, REPORT_DIR)
        with open(report_path) as f:
            html = f.read()
        st.components.v1.html(html, height=700, scrolling=True)