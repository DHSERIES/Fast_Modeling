
import streamlit as st
import pandas as pd
import os
# Title


def show_DataPrep():
    st.title("🔧 Data Preparation for Training")

    # Load dataset (from session state or upload)
    # get the uploaded or selected dataset path
    path = st.session_state.get("dataset_path", None)
    if not path or not os.path.exists(path):
        st.warning("Please upload or select a dataset on the Home / Import Data page first.")
        st.stop() 

    # show a small preview
    df = pd.read_csv(path)


    target = st.selectbox("Select Target (Label) Column", options=[None] + df.columns.tolist(), key="target_column_select")
    if target:
        st.session_state["target_column"] = target
    else:
        st.warning("Please select a target column to enable training.")
    
    # Identify feature columns by type
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    col1, col2 = st.columns([1, 1])

    # --- Column 1: Missing Value Handling per column ---
    
    with col1:
        st.subheader("Missing Value Handling")
        auto_missing = st.checkbox("Auto-process missing for all columns", key="auto_missing")
        missing_drop = []
        missing_impute = {}
        if auto_missing:
            # default: impute numeric with mean, categorical with most_frequent
            for col in numeric_cols:
                missing_impute[col] = ("mean", None)
            for col in categorical_cols:
                missing_impute[col] = ("most_frequent", None)
        else:
            for col in numeric_cols + categorical_cols:
                with st.expander(f"{col}"):
                    method = st.selectbox(
                        "Method", ["None", "Drop Rows", "Impute"], key=f"mv_method_{col}"
                    )
                    if method == "Drop Rows":
                        missing_drop.append(col)
                    elif method == "Impute":
                        strategy = st.selectbox(
                            "Strategy", ["mean", "median", "most_frequent", "constant"], key=f"mv_strat_{col}"
                        )
                        fill = None
                        if strategy == "constant":
                            fill = st.text_input("Constant value", value="0", key=f"mv_fill_{col}")
                        missing_impute[col] = (strategy, fill)

    # --- Column 2: Scaling & Encoding per column ---
    with col2:
        st.subheader("Scaling & Encoding")
        auto_preproc = st.checkbox("Auto-scale and encode all columns", key="auto_preproc")
        scale_cols = []
        encode_cols = {}
        if auto_preproc:
            scale_cols = numeric_cols.copy()
            # default encode all categoricals with OneHot
            encode_cols = {col: "OneHot" for col in categorical_cols}
        else:
            for col in numeric_cols:
                with st.expander(f"Scale: {col}"):
                    do_scale = st.checkbox("Scale this column", key=f"scale_{col}")
                    if do_scale:
                        scale_cols.append(col)
            for col in categorical_cols:
                with st.expander(f"Encode: {col}"):
                    do_encode = st.checkbox("Encode this column", key=f"encode_{col}")
                    if do_encode:
                        method = st.selectbox(
                            "Method", ["OneHot", "Ordinal"], key=f"enc_method_{col}"
                        )
                        encode_cols[col] = method


        # Apply settings when button is clicked
    if st.button("Apply Settings"):
        # Save selected target and pipeline settings to session state
        st.session_state["target_column"] = target
        st.session_state["missing_drop"] = missing_drop
        st.session_state["missing_impute"] = missing_impute
        st.session_state["scale_columns"] = scale_cols
        st.session_state["encode_columns"] = encode_cols
            
        # Check if target column and other required fields are not None
        if all([st.session_state['target_column'] is not None, 
                st.session_state['missing_drop'] is not None,
                st.session_state['missing_impute'] is not None,
                st.session_state['scale_columns'] is not None,
                st.session_state['encode_columns'] is not None,
                st.session_state['target_column'] is not None]):
            st.success("Pipeline settings and label selection applied successfully!")
        else:
            st.error("Please ensure all settings are selected before applying.")