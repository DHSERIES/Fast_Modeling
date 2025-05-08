import streamlit as st
from ImportData  import show_import_data
from EDA         import show_eda
from DataPrep    import show_DataPrep
from ModelSelect import show_modelselect
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Import Data", "EDA" , "DataPrep", "ModelSelect"])

def show_home():
        # — 1) Header & description —
    st.title("🏠 Home")
    st.markdown(
        """
        **Welcome!**  
        This interactive app helps you:
        - 🚀 Import and explore your data  
        - 📊 Perform automated EDA  
        - 🧹 Prep and clean for modeling
        - Chosing model and paramter for training 
        
        Use the sidebar to navigate between steps.
        """
    )
    st.write("---")

    # — 5) Tips & next steps —
    with st.expander("ℹ️ How to use this app"):
        st.markdown(
            """
            1. Go to **Import Data** to upload a CSV, Excel or connect to a database.  
            2. Switch to **EDA** to see distributions, correlations, and summary statistics.  
            3. Use **DataPrep** to handle missing values, encode categoricals, and scale features.  
            4. Return here anytime to monitor metrics or jump between steps.
            """
        )

if page == "Home":
    show_home()
elif page == "Import Data":
    show_import_data()
elif page == "EDA":
    show_eda()
elif page == "DataPrep":
    show_DataPrep()
elif page == "ModelSelect":
    show_modelselect()
