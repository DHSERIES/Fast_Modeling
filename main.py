import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Importing Plotly Express as 'px'
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

            
def EDA():
    options = ["iris_example.csv"]  # default dataset
    if "all_dataset" in st.session_state:
        options.append("Uploaded dataset")

    # Create a selectbox to choose the dataset
    option = st.selectbox(
        "Select dataset:",
        options
    )
    
    st.write("You selected:", option)
    df = pd.read_csv(f"{option}")
    st.write(option)
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

def contact_page():
    st.title("Contact Page")
    st.write("This is the contact page. Get in touch with us!")

# Create sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "About", "Contact"])

# Conditional rendering based on the page selected
if page == "Home":
    home_page()
elif page == "About":
    EDA()
elif page == "Contact":
    contact_page()