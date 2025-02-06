import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ----------------------------
# Page Title and Description
# ----------------------------
st.title("Data Preprocessing")
st.write("""
This page allows you to preprocess your dataset before moving on to Exploratory Data Analysis (EDA). 
You can handle missing values, remove duplicates, encode categorical features, and scale numerical features.
""")

# ----------------------------
# Dataset Upload / Retrieval
# ----------------------------
# If the dataset is not already in session_state, allow the user to upload it.
def PREPROCESSING(dataset = 'iris_example.csv'):

    df = pd.read_csv(dataset)

    # Display a preview of the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ----------------------------
    # 1. Handling Missing Values
    # ----------------------------
    st.header("1. Handle Missing Values")
    missing_option = st.selectbox(
        "Select a method to handle missing values:",
        ("None", "Drop rows", "Fill with Mean", "Fill with Median", "Fill with Mode")
    )

    if missing_option != "None":
        if st.button("Apply Missing Value Handling"):
            df_pre = df.copy()  # Copy the dataframe for preprocessing
            if missing_option == "Drop rows":
                df_pre = df_pre.dropna()
            elif missing_option == "Fill with Mean":
                for col in df_pre.select_dtypes(include=[np.number]).columns:
                    df_pre[col].fillna(df_pre[col].mean(), inplace=True)
            elif missing_option == "Fill with Median":
                for col in df_pre.select_dtypes(include=[np.number]).columns:
                    df_pre[col].fillna(df_pre[col].median(), inplace=True)
            elif missing_option == "Fill with Mode":
                for col in df_pre.columns:
                    df_pre[col].fillna(df_pre[col].mode()[0], inplace=True)
            df = df_pre  # Update the main dataframe
            st.session_state.df = df
            st.success(f"Missing values handled using '{missing_option}'.")
            st.dataframe(df.head())

    # ----------------------------
    # 2. Removing Duplicates
    # ----------------------------
    st.header("2. Remove Duplicate Rows")
    if st.button("Remove Duplicates"):
        df = df.drop_duplicates()
        st.session_state.df = df
        st.success("Duplicate rows removed.")
        st.dataframe(df.head())

    # ----------------------------
    # 3. Encoding Categorical Variables
    # ----------------------------
    st.header("3. Encode Categorical Features")
    encoding_option = st.selectbox(
        "Select an encoding method for categorical columns:",
        ("None", "Label Encoding", "One-Hot Encoding")
    )

    if encoding_option != "None":
        if st.button("Apply Encoding"):
            df_pre = df.copy()  # Work on a copy for safe processing
            # Identify categorical columns (here, we assume object type as categorical)
            cat_cols = df_pre.select_dtypes(include=["object"]).columns.tolist()
            if not cat_cols:
                st.info("No categorical columns detected in the dataset.")
            else:
                if encoding_option == "Label Encoding":
                    le = LabelEncoder()
                    for col in cat_cols:
                        try:
                            df_pre[col] = le.fit_transform(df_pre[col])
                        except Exception as e:
                            st.error(f"Error encoding column '{col}': {e}")
                elif encoding_option == "One-Hot Encoding":
                    df_pre = pd.get_dummies(df_pre, columns=cat_cols)
                df = df_pre
                st.session_state.df = df
                st.success(f"Categorical features encoded using '{encoding_option}'.")
                st.dataframe(df.head())

    # ----------------------------
    # 4. Scaling Numerical Features
    # ----------------------------
    st.header("4. Scale Numerical Features")
    scaling_option = st.selectbox(
        "Select a scaling method for numerical columns:",
        ("None", "Standard Scaler", "Min-Max Scaler")
    )

    if scaling_option != "None":
        if st.button("Apply Scaling"):
            df_pre = df.copy()
            num_cols = df_pre.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.info("No numerical columns detected for scaling.")
            else:
                try:
                    if scaling_option == "Standard Scaler":
                        scaler = StandardScaler()
                    elif scaling_option == "Min-Max Scaler":
                        scaler = MinMaxScaler()
                    df_pre[num_cols] = scaler.fit_transform(df_pre[num_cols])
                    df = df_pre
                    st.session_state.df = df
                    st.success(f"Numerical features scaled using '{scaling_option}'.")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error during scaling: {e}")

    # ----------------------------
    # End of Preprocessing Page
    # ----------------------------
    st.write("Preprocessing complete! The modified dataset is stored in session_state and is ready for EDA.")