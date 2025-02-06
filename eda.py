import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Importing Plotly Express as 'px'

def EDA(path = "iris_example.csv"):
    # Initialize session state variable if it doesn't exist
    df = pd.read_csv(path)

    st.dataframe(df.head())
    # Display basic dataset info
    st.subheader("Basic Information")
    st.write(f"Shape of the dataset: {df.shape}")
    st.write(f"Columns in the dataset: {df.columns}")
    st.write(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Histograms for numerical columns
    st.subheader("Histograms for Numerical Features")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        st.write(f"Histogram for {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Boxplot for detecting outliers
    st.subheader("Boxplots for Outlier Detection")
    for col in num_cols:
        st.write(f"Boxplot for {col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)
    
    # Pairplot for feature relationships (if dataset is small)
    st.subheader("Pairplot of Features")
    if len(df) <= 500:  # Avoiding too large pairplot
        st.write("Pairplot (first 500 rows)")
        sns.pairplot(df)
        st.pyplot()
    else:
        st.write("Dataset too large for pairplot. Displaying first 500 rows pairplot.")
        sns.pairplot(df.head(500))
        st.pyplot()
    
    # Plotting a feature distribution with Plotly for interactivity
    st.subheader("Interactive Feature Distribution")
    feature = st.selectbox("Select feature for interactive plot", num_cols)
    if feature:
        fig = px.histogram(df, x=feature, nbins=20, title=f"Distribution of {feature}")
        st.plotly_chart(fig)