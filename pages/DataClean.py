
import streamlit as st
import pandas as pd
import os
# Title
def _store_impute():
    st.session_state["impute_method"] = st.session_state["_impute_method"]
    st.session_state["_impute_method"] = st.session_state["impute_method"]

def _get_impute(default=None):
    return st.session_state.get("_impute_method", default)

st.title("🔧 Data Preparation for Training")

# Load dataset (from session state or upload)
# get the uploaded or selected dataset path
path = st.session_state.get("dataset_path", None)
if not path or not os.path.exists(path):
    st.warning("Please upload or select a dataset on the Home / Import Data page first.")
    st.stop() 

# show a small preview
df = pd.read_csv(path)

st.subheader("Missing Value Handling")
if df.isna().sum().sum() > 0:
    impute_method = st.selectbox(
        "Select Impute Method",
        ("dropall", "fillall", "Custom"),
        key="_impute_method",
        on_change=_store_impute
    )

    st.write(f"You selected: {impute_method}")
else:
    st.write("No missing data")

# Options for fillall
if impute_method == "fillall":
    fill_strategy = st.selectbox(
        "Fillall Strategy",
        ("mode", "mean", "median"),
        key="_fill_strategy"
    )
# Options for Custom
if impute_method == "Custom":
    custom_strategy = st.selectbox(
        "Select Custom Imputation",
        ("interpolate", "ffill", "bfill"),
        key="_custom_strategy"
    )
if st.button("Apply Missing Value Handling"):
    if impute_method == 'dropall':
        cleaned_df = df.dropna()
        st.success("Dropped all rows with missing values.")

    elif impute_method == 'fillall':
        if fill_strategy == "mode":
            mode_vals = df.mode().iloc[0]
            cleaned_df = df.fillna(mode_vals)
            st.success("Filled missing values with column modes.")
        elif fill_strategy == "mean":
            cleaned_df = df.fillna(df.mean(numeric_only=True))
            st.success("Filled numeric missing with means.")
        elif fill_strategy == "median":
            cleaned_df = df.fillna(df.median(numeric_only=True))
            st.success("Filled numeric missing with medians.")

    elif impute_method == 'Custom':
        if custom_strategy == "interpolate":
            cleaned_df = df.interpolate()
            st.success("Imputed missing values by interpolation.")
        elif custom_strategy == "ffill":
            cleaned_df = df.fillna(method='ffill')
            st.success("Imputed missing values with forward fill.")
        elif custom_strategy == "bfill":
            cleaned_df = df.fillna(method='bfill')
            st.success("Imputed missing values with backward fill.")



    st.subheader("Cleaned Data Preview (First 5 rows)")
    st.dataframe(cleaned_df.head())