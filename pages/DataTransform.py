
import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# Title
def _store_target():
    st.session_state["target_column"] = st.session_state["_target_column"]
    st.session_state["_target_column"] = st.session_state["target_column"]

def auto_scale_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically transforms the DataFrame:
    - Standardizes numeric columns (Z-score).
    - One-hot encodes categorical columns.
    
    Returns transformed DataFrame.
    """
    df = df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Scale numeric
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # One-hot encode categorical
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    return df

def apply_feature_transform(
    df: pd.DataFrame,
    selected_feature: str,
    method: str,
) -> pd.DataFrame:
    """
    Detects the type of `selected_feature` in `df` and applies:
      - numerical: StandardScaler or MinMaxScaler
      - categorical: one-hot, label-encode, or (stub) embedding

    Returns the modified DataFrame (with the feature column replaced, or expanded
    in the case of one-hot).
    """
    if selected_feature not in df.columns:
        raise ValueError(f"Feature '{selected_feature}' not found in DataFrame.")

    series = df[selected_feature]
    
    # --- Numerical ---
    if pd.api.types.is_numeric_dtype(series):
        X = series.values.reshape(-1, 1)
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown numeric_transform: {method}")
        
        df[selected_feature] = scaler.fit_transform(X)
        return df

    # --- Categorical ---
    elif isinstance('categories', pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
        if method == "onehot":
            # get_dummies returns a new DataFrame
            dummies = pd.get_dummies(df[selected_feature], prefix=selected_feature)
            # drop original column, concat dummies
            df = df.drop(columns=[selected_feature])
            df = pd.concat([df, dummies], axis=1)
            return df

        elif method == "label":
            le = LabelEncoder()
            df[selected_feature] = le.fit_transform(series.astype(str))
            return df


    else:
        raise TypeError(f"Cannot determine how to process dtype {series.dtype} for '{selected_feature}'.")

def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Drops the specified columns from the DataFrame.
    
    Parameters:
    - df: Input DataFrame
    - columns_to_drop: List of column names to drop
    
    Returns:
    - DataFrame with specified columns removed
    """
    df = df.copy()
    missing = [col for col in columns_to_drop if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    return df.drop(columns=columns_to_drop)


st.title("🔧 Data Preparation for Training")

# Load dataset (from session state or upload)
# get the uploaded or selected dataset path
path = st.session_state.get("dataset_path", None)
if not path or not os.path.exists(path):
    st.warning("Please upload or select a dataset on the Home / Import Data page first.")
    st.stop() 

# show a small preview
df = pd.read_csv(path)

# st.write(f"you have selected {st.session_state['_impute_method']} cleaned method ")
st.selectbox(
    "Select Target (Label) Column",
    options=[None] + df.columns.tolist(),
    key="_target_column",
    on_change=_store_target
)

# Now you can safely use st.session_state["target_column"] anywhere,
# and it will retain its value when you switch pages.
if st.session_state["_target_column"] is None:
    st.warning("Please select a target column to enable training.")
    st.stop()

target = st.session_state["_target_column"]

cols_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())

if st.button("Auto Scale All Features"):
    try:
        df = auto_scale_dataframe(df)
        st.success("All numeric features standardized and categorical features one‑hot encoded.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error during auto-scaling: {e}")


# 5. Per‑feature transform
all_features = [col for col in df.columns if col != target]
selected_feature = st.selectbox("Select feature column", options=all_features)

# Determine numeric vs. categorical
numeric_cols = df.select_dtypes(include="number").columns.tolist()
is_numeric = selected_feature in numeric_cols

# Show the appropriate transform selector
if is_numeric:
    numeric_transform = st.selectbox("For numerical features →", ["standard (Z‑score)", "minmax (0–1)"])
else:
    categorical_transform = st.selectbox("For categorical features →", ["one‑hot", "label"])

# Map UI labels to your internal method names
numeric_map = {"standard (Z‑score)": "standard", "minmax (0–1)": "minmax"}
cat_map     = {"one‑hot": "onehot",           "label":    "label"}

if st.button("Apply selected feature transform"):
    try:
        if is_numeric:
            method = numeric_map[numeric_transform]
        else:
            method = cat_map[categorical_transform]

        df = apply_feature_transform(df, selected_feature, method)
        st.success(f"Applied {'numeric' if is_numeric else 'categorical'} transform: `{method}`.")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error: {e}")