import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
def read_dataset(dataset_choice):
    if dataset_choice != "-- Select a dataset --":
        url = datasets[dataset_choice]
        try:
            df = pd.read_csv(url)
            st.success(f"✅ Successfully loaded {dataset_choice} dataset.")
            return df
        except Exception as e:
            st.error(f"⚠️ Failed to load dataset: {e}")
            return None
    else:
        st.info("👉 Please select a dataset from the sidebar.")
        return None


# SIDE PAGE
datasets = {}

data_path = "data/"
# Loop through each file in the specified directory
for filename in os.listdir(data_path):
    file_path = os.path.join(data_path, filename)
    if os.path.isfile(file_path):  # Ensure it's a file and not a directory
        datasets[filename] = file_path  # Use the filename as the key

# Create a Streamlit sidebar selectbox to choose a dataset
dataset_choice = st.sidebar.selectbox(
    "Select a dataset:",
    ["-- Select a dataset --"] + list(datasets.keys())
)

page = st.sidebar.radio("Navigate to:", ["Home", "PREPROCESS", "EDA", "MODELING"])

# Home Page
if page == "Home":
    st.title("🏠 Home Page")
    DATA_DIR = "data/"

    st.header("Import File")
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head(5))

    # IMPORT DATASET
    if st.button("Import dataset "):
        filename = uploaded_file.name
        output_file_path = os.path.join(data_path, filename)
        df.to_csv(output_file_path, index=False)
        

# Page 2
elif page == "PREPROCESS":
    st.title("📄 Page 2")
    st.write("This is Page 2 content.")
    df = read_dataset(dataset_choice)
    if df is not None:
        pass
    

# Page 3
elif page == "EDA":
    st.title("📄 Page 3")
    st.write("This is Page 3 content.")
    df = read_dataset(dataset_choice)
    if df is not None:
        pass

elif page == "MODELING":
    st.title("📄 Page 4")
    st.write("This is Page 4 content.")
    df = read_dataset(dataset_choice)
    if df is not None:
        pass