import streamlit as st
import pandas as pd

def load_data(file):
    """Load CSV file into a DataFrame."""
    df = pd.read_csv(file)
    return df

def display_preprocessing_steps(df):
    """Display fast analysis including info, missing values, and skewness."""
    st.subheader("Data Analysis")
    st.write("**Shape of the Data:**", df.shape)
    st.write("**First 5 Rows:**")
    st.dataframe(df.head())
    
    st.write("**Summary Statistics:**")
    st.dataframe(df.describe())
    
    st.write("**Missing Values per Column:**")
    st.dataframe(df.isnull().sum().to_frame(name='Missing Count'))
    
    # Calculate skewness only for numeric columns.
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.write("**Skewness (Numeric Columns):**")
        skewness = df[numeric_cols].skew().to_frame(name='Skewness')
        st.dataframe(skewness)
    

