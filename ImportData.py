# importdata.py
import streamlit as st
import pandas as pd
import os

DATA_DIR = "data/"

def show_import_data():
    st.title("📥 Import Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        if st.button("Save to data/"):
            os.makedirs(DATA_DIR, exist_ok=True)
            path = os.path.join(DATA_DIR, uploaded_file.name)
            df.to_csv(path, index=False)
            st.success(f"Saved to `{path}`")
            # store path so other pages can see it
            st.session_state["dataset_path"] = path

    # list CSVs in data/
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    choice = st.selectbox("Select a dataset", ["--"] + files)

    if choice != "--":
        path = os.path.join(DATA_DIR, choice)
        st.session_state["dataset_path"] = path
        st.success(f"Selected `{choice}`")
        df = pd.read_csv(path)
        st.write(df.head())