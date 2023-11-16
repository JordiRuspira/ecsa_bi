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
from shroomdk import ShroomDK

st.cache(suppress_st_warning=True)  
st.set_page_config(page_title="ECSA BI Dashboard", layout="wide",initial_sidebar_state="collapsed")

flipside_key = st.secrets["API_KEY"]
sdk = ShroomDK(flipside_key)
# Query Flipside using their Python SDK
def query_flipside(q):
    sdk = ShroomDK(flipside_key)
    result_list = []
    for i in range(1, 11):  # max is a million rows @ 100k per page
        data = sdk.query(q, page_size=100000, page_number=i)
        if data.run_stats.record_count == 0:
            break
        else:
            result_list.append(data.records)
    result_df = pd.DataFrame()
    for idx, each_list in enumerate(result_list):
        if idx == 0:
            result_df = pd.json_normalize(each_list)
        else:
            try:
                result_df = pd.concat([result_df, pd.json_normalize(each_list)])
            except:
                continue
    result_df.drop(columns=["__row_index"], inplace=True)
    return result_df


# In[2]:
st.title('Introduction')

# In[3]:
st.markdown('This is a first version of a dashboard which will evolve over time and anyone can fork at any given time.')
st.markdown('The end goal of a first version would ideally be structured as follows:')
st.write('- First, number of offers.')
st.write('- Number of staked offers (become ecsa operations).')
st.write('- Deadlines met for tasks within offers.')
st.write('- Weekly hour limit reached but not breached.')
 

tab1, tab2 = st.tabs(["Section 1 - Number of offers","Section 2 "])

# In[7]:

with tab1:
    
    
    st.subheader("Number of offers")
    st.write('')
    st.write('At the start of the offer market, each participant was asked to fill a link with a few information: name, interests, skills, projected weekly hours. That data has been collected and uploaded as a csv here.')
    st.write('')
    
    df_initial_data = pd.read_csv('offer_markets_initial_hours.csv')
    fig = px.pie(df_initial_data, values='Q4_weekly_hours', names='Node Affiliation', title='Q4 Projected weekly hours by node affiliation',
                hover_data='Q4_weekly_hours', labels={'Q4_weekly_hours':'Q4 Weekly hours'})
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.write('Remember that each of us wrote our interests? Well, a cool feature we can look at is a wordcloud of our interests, which is shown below. Unfortunately, adding it on streamlit has not been possible so it is an ad-hoc picture straight from a python code, which it is also accessible through the github of this project.')
    st.write('')
    
    st.image("img/wordcloud_ecsa.png")
    
with tab2:
    
    st.subheader('Test2')
    st.write('')
    st.write('Test2 text')
    
    sql1 = """
       select date_trunc('day', block_timestamp) as date,
    count(distinct tx_hash) as num_mints,
count(distinct nft_to_address) as unique_minters,
sum(num_mints) over (order by date) as cumulative_mints,
sum(unique_minters)  over (order by date) as cumulative_minters from polygon.nft.ez_nft_mints
where nft_address = '0xb19e16fa7bfa2924a17b77d0379ff6e2899e8397'
group by 1 '  
    """
    
    st.experimental_memo(ttl=1000000)
    @st.experimental_memo
    def compute(a):
        results=sdk.query(a)
        return results
    
    results1 = compute(sql1)
    df1 = pd.DataFrame(results1.records)
    
    fig1 = px.bar(df1, x="date", y="num_mints", color_discrete_sequence=px.colors.qualitative.Pastel2)
    fig1.update_layout(
    title='Daily ECSA NFTs minted',
    xaxis_tickfont_size=14,
    yaxis_tickfont_size=14,
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
