import streamlit as st
from ImportData  import show_import_data
from EDA         import show_eda

st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Import Data", "EDA"])

def show_home():
    st.title("🏠 Home")

if page == "Home":
    show_home()
elif page == "Import Data":
    show_import_data()
elif page == "EDA":
    show_eda()