import streamlit as st
import pandas as pd

def home_page():
    st.header("Import File")
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file (CSV, Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Display the file name
        st.write(f"File uploaded: {uploaded_file.name}")
        
        # Reading and displaying the file content
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, index_col= None)
        df.to_csv("source_data.csv",index = None)
        # Show the first few rows of the dataframe
        st.write("Preview of the dataset:")
        st.dataframe(df.head())

        profile_button = st.button("Generate Report")
 

        if profile_button: 
            st.title("Exploratory Data Analysis")

def about_page():
    st.title("About Page")
    st.write("This is the About Page. Learn more about the application here.")

def contact_page():
    st.title("Contact Page")
    st.write("This is the contact page. Get in touch with us!")

# Create sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "About", "Contact"])

# Conditional rendering based on the page selected
if page == "Home":
    home_page()
elif page == "About":
    about_page()
elif page == "Contact":
    contact_page()