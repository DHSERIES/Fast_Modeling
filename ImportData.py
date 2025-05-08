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
        rows, cols = df.shape
        missing = df.isna().mean().mean() * 100
        num_numeric = len(df.select_dtypes(include="number").columns)
        num_categorical = len(df.select_dtypes(exclude="number").columns)
    else:
        rows = cols = num_numeric = num_categorical = 0
        missing = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Missing %", f"{missing:.1f}%")
    c4.metric("Numeric / Cat.", f"{num_numeric} / {num_categorical}")