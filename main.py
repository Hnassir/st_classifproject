import streamlit as st
from EDA import app as eda_app
from classif import app as class_app



with st.sidebar :

    st.title('data analysis')
    page=st.radio('select a page',options=['EDA','Classification'])

if page =='EDA' :
    eda_app()
else:
    class_app()