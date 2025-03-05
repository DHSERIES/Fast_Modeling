import streamlit as st

def app(df):
    st.header("Process file")
    st.write(df)

    st.button('feature engineering')
    