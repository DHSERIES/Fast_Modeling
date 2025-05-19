import streamlit as st

pages = [
    st.Page("pages/ImportData.py", title="Import Data", icon="📥"),
    st.Page("pages/AnalysisData.py", title="EDA", icon="📊"),
    st.Page("pages/DataClean.py", title="Data Clean", icon="🧹"),
    st.Page("pages/DataTransform.py", title="Transform", icon="🔄"),
    st.Page("pages/ModelSelect.py", title="Model Select", icon="🤖"),
]

# renders sidebar, returns chosen page object
selected = st.navigation(pages, position="sidebar")
selected.run()

