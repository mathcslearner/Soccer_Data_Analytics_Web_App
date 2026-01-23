# League comparison page for the Streamlit Soccer Data Analytics App

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
st.title("European Top 5 Leagues - League Comparison Analytics")
st.markdown(
    "Head-to-head analysis using advanced metrics for two leagues"
)

st.divider()