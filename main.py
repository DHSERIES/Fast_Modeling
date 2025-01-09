import streamlit as st
import pandas as pd
import sweetviz as sv
from streamlit_pandas_profiling import st_profile_report

# Customize the sidebar
st.sidebar.title("Machine Learning Dashboard")
st.sidebar.markdown("### Choose an Option:")

# Sidebar buttons 
import_file = st.sidebar.button("🔽 Import File")
process_data = st.sidebar.button("🔄 Processing Data")
train_model = st.sidebar.button("⚙️ Model Train")
monitor = st.sidebar.button("📊 Monitor")

# Main content based on button selection
if import_file:
    st.header("Import File")
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file (CSV, Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Display the file name
        st.write(f"File uploaded: {uploaded_file.name}")
        
        # Reading and displaying the file content
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file,index_col= None)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, index_col= None)
        df.to_csv("source_data.csv",index = None)
        # Show the first few rows of the dataframe
        st.write("Preview of the dataset:")
        st.dataframe(df.head())

        profile_button = st.button("Generate Report")
 

        if profile_button: 
            st.title("Exploratory Data Analysis")



if process_data:
    st.header("Processing Data")
    st.write("Here you can process and clean your data.")

if train_model:
    st.header("Model Training")
    st.write("Here you can train your machine learning model.")

if monitor:
    st.header("Monitor")
    st.write("Here you can monitor model performance or other metrics.")