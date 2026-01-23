#Goalkeeping page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------
# Page config
# --------------------
st.set_page_config(
    page_title="European Soccer Analytics",
    layout="wide"
)

# --------------------
# Load data
# --------------------
@st.cache_data
def load_data():
    return pd.read_csv("team_stats_2024-2025.csv")

df = load_data()

# --------------------
# Title
# --------------------
st.title("European Top 5 Leagues - Goalkeeping Analytics")
st.markdown(
    "Interactive analysis of defensive metrics across European top 5 leagues"
)

st.divider()