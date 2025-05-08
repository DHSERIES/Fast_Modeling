# eda.py
import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components


st.set_page_config(page_title="Gallery Page", page_icon=":camera:")
def show_eda():
    st.title("📊 Full‑Page EDA Report")

    # get the uploaded or selected dataset path
    path = st.session_state.get("dataset_path", None)
    
    if not path or not os.path.exists(path):
        st.warning("Please upload or select a dataset on the Home / Import Data page first.")
        st.stop()

    # show a small preview
    df = pd.read_csv(path)
    st.write("Preview of your data:", df.head(3))

    # when clicked, generate & embed the full report
    if st.button("Generate Full‑Page Profiling Report"):
        with st.spinner("Generating profiling report…"):
            pr = ProfileReport(df, explorative=True)

            # 2) convert to HTML and embed as a full‑page scrollable component
            html = pr.to_html()
            components.html(
                html,
                height=st.session_state.get("report_height", 1000), 
                scrolling=True,
            )