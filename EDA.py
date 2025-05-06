# eda.py
import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

REPORT_DIR = "report/"

@st.cache_data
def show_eda():
    st.title("📊 EDA with Sweetviz")

    path = st.session_state.get("dataset_path", None)
    if not path:
        st.warning("Please upload or select a dataset on the Home / Import Data page first.")
        return

    df = pd.read_csv(path)
    st.write("Preview:", df.head())

    if st.button("Generate Profiling Report"):
        with st.spinner("Generating..."):
            pr = ProfileReport(df, explorative=True)
            st_profile_report(pr)