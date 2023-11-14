# streamlit_app.py

import hmac
import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

# Call the check_password function
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here
import datetime
import streamlit as st
import pandas as pd
import requests
import json
import time
import plotly.graph_objects as go
import random
import plotly.io as pio
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.ticker as ticker
import numpy as np
import plotly.express as px 

st.cache(suppress_st_warning=True)  
st.set_page_config(page_title="ECSA BI Dashboard", layout="wide",initial_sidebar_state="collapsed")

# Rest of the script...

# In[2]:
st.title('Introduction')

# In[3]:
st.markdown('This is a first version of a dashboard which will evolve over time and anyone can fork at any given time.')
st.markdown('It is structured as follows:')
st.write('- First, number of offers.')
st.write('- number of staked offers (become ecsa operations).')
st.write('- deadlines met for tasks within offers.')
st.write('- weekly hour limit reached but not breached.')
st.markdown('If there`s any missing information or misleading one, please do not hesitate to reach out on twitter.')

 

tab1, tab2 = st.tabs(["Fake data test","Input data here"])

# In[7]:

with tab1:
    
    
    st.subheader("Test1")
    st.write('')
    st.write('Test1 text.')
    st.write('')
    
    df_initial_data = pd.read_csv('offer_markets_initial_hours.csv')
    fig = px.pie(df_initial_data, values='Q4_weekly_hours', names='Node Affiliation', title='Q4 Projected weekly hours by node affiliation')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    st.image("img/wordcloud_ecsa.png")
    
with tab2:
    
    st.subheader('Test2')
    st.write('')
    st.write('Test2 text')
