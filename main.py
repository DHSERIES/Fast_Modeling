import streamlit as st
from ImportData  import show_import_data
from EDA         import show_eda
from DataPrep    import show_DataPrep

st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Import Data", "EDA" , "DataPrep"])

def show_home():
    st.title("🏠 Home")

if page == "Home":
    show_home()
elif page == "Import Data":
    show_import_data()
elif page == "EDA":
    show_eda()
elif page == "DataPrep":
    show_DataPrep()