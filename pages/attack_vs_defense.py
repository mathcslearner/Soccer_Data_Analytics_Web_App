#Attack vs Defense page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Attack vs Defense", layout="wide")

st.title("⚔️ Attack vs Defense — Tactical Identity")
st.markdown(
    """
    This page maps teams by **attacking strength** and **defensive identity**.
    Defense is split into **pressing behavior** and **shot prevention quality**,
    then combined into a single axis for clustering.
    """
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("team_stats_2024-2025.csv")

df = load_data()

# --------------------------------------------------
# Basic prep
# --------------------------------------------------
df = df.copy()

# Opponent possession proxy
df["Opp_Poss"] = 100 - df["Poss"]

scaler = StandardScaler()

# --------------------------------------------------
# ATTACK INDEX
# --------------------------------------------------
attack_features = [
    "xG",
    "Shots",
    "Touches Att 3rd",
    "SCA",
    "GCA"
]

df[attack_features] = scaler.fit_transform(df[attack_features])
df["Attack_Index"] = df[attack_features].mean(axis=1)

# --------------------------------------------------
# PRESSING / DEFENSIVE BEHAVIOR 
# --------------------------------------------------
df["Tackles_per_opp_poss"] = df["Tackles"] / df["Opp_Poss"]
df["Interceptions_per_opp_poss"] = df["Interceptions"] / df["Opp_Poss"]
df["Att3_Tackles_per_opp_poss"] = df["Tackles Att 3rd"] / df["Opp_Poss"]

pressing_features = [
    "Tackles_per_opp_poss",
    "Interceptions_per_opp_poss",
    "Att3_Tackles_per_opp_poss",
    "Recoveries",
    "Dribble Stops"
]

df[pressing_features] = scaler.fit_transform(df[pressing_features])
df["Pressing_Index"] = df[pressing_features].mean(axis=1)

# --------------------------------------------------
# DEFENSIVE OUTCOMES (SOLIDITY)
# lower = better → invert
# --------------------------------------------------
df["xGA_inv"] = -df["xG Allowed"]
df["GA_inv"] = -df["Goals Allowed"]
df["SoTA_inv"] = -df["Shots on Target Allowed"]
df["Errors_inv"] = -df["Errors"]

defensive_outcome_features = [
    "xGA_inv",
    "GA_inv",
    "SoTA_inv",
    "Errors_inv"
]

df[defensive_outcome_features] = scaler.fit_transform(
    df[defensive_outcome_features]
)

df["Defensive_Solidity"] = df[defensive_outcome_features].mean(axis=1)

# --------------------------------------------------
# FINAL DEFENSE INDEX (STYLE + OUTCOME)
# --------------------------------------------------
df["Defense_Index"] = (
    0.5 * df["Pressing_Index"] +
    0.5 * df["Defensive_Solidity"]
)

# --------------------------------------------------
# Clustering
# --------------------------------------------------
st.sidebar.header("Clustering")

k = st.sidebar.slider("Number of playstyle clusters", 2, 8, 4)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(
    df[["Attack_Index", "Defense_Index"]]
)

# --------------------------------------------------
# Scatter plot (fixed axis ranges)
# --------------------------------------------------

axis_limit = 4

fig = px.scatter(
    df,
    x="Attack_Index",
    y="Defense_Index",
    color="Cluster",
    hover_name="team",
    title="Team Tactical Identity: Attack vs Defense",
    labels={
        "Attack_Index": "Attacking Strength (z-score)",
        "Defense_Index": "Defensive Strength (z-score)"
    }
)

# Zero axes
fig.add_hline(y=0, line_width=2, line_color="white")
fig.add_vline(x=0, line_width=2, line_color="white")

# Force fixed ranges
fig.update_xaxes(range=[-axis_limit, axis_limit], fixedrange=True)
fig.update_yaxes(range=[-axis_limit, axis_limit], fixedrange=True)

# Force square aspect ratio (without expanding ranges)
fig.update_layout(
    yaxis=dict(scaleanchor="x", scaleratio=1),
    margin=dict(l=40, r=40, t=60, b=40),
    height=650
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Tactical guide
# --------------------------------------------------
st.subheader("🧭 How to Read This Map")

st.markdown(
    """
    **Top-right** → Dominant teams (strong attack + strong defense)  
    **Bottom-right** → Aggressive but fragile (chaotic, transition-heavy)  
    **Top-left** → Low-block efficiency (defend well, limited attack)  
    **Bottom-left** → Struggling teams  

    Defensive strength blends **pressing intent** and **shot prevention quality**.
    """
)
