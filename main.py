from home_page import home_page
from working_page import working_page
import streamlit as st



# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'



# Page router
if st.session_state.page == 'home':
    home_page()

elif st.session_state.page == 'working':
    working_page()



