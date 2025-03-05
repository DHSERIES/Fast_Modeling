import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def home_page():
    st.header("Import File")
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file (CSV, Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Display the file name
        st.write(f"File uploaded: {uploaded_file.name}")
        
        # Read and display the file content
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        
        # Store the uploaded file in session state if needed
        st.session_state.uploaded_file = uploaded_file

        # Display the DataFrame
        st.write(df)
        st.dataframe(df.head())
