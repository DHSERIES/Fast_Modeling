import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from preprocessing import display_preprocessing_steps
DATA_DIR = "data/"
st.session_state["selected_table"] = pd.read_csv('data/iris_example.csv')

st.header("Import File")
# File uploader widget
uploaded_file = st.file_uploader("Choose a file (CSV, Excel)", type=["csv", "xlsx"])

dataset_files = {
    f: os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
}
if uploaded_file is not None:
    # Display the file name
    st.write(f"File uploaded: {uploaded_file.name}")
    
    # Read and display the file content

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raise TypeError("File unreadble")
    
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    dataset_files = {
        f: os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
    }
    st.success(f"Imported dataset saved to {file_path}")

# Create a drop-down list for dataset selection
selected_dataset = st.selectbox("Choose a dataset", list(dataset_files.keys()))

if st.button("Select Dataset"):
    # Read the selected dataset
    df = pd.read_csv(dataset_files[selected_dataset])
    st.write(f"Displaying the {selected_dataset}:")
    st.dataframe(df)
    copy = df
    display_preprocessing_steps(copy)

