#Goalkeeping page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Goalkeeping & Defense", layout="wide")

st.title("🧤 Goalkeeping & Defensive Analysis")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("team_stats_2024-2025.csv")

df = load_data()

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filters")

league = st.sidebar.selectbox(
    "League",
    sorted(df["league"].unique())
)

season = st.sidebar.selectbox(
    "Season",
    sorted(df[df["league"] == league]["season"].unique())
)

filtered_df = df[
    (df["league"] == league) &
    (df["season"] == season)
]

# --------------------------------------------------
# Derived metrics
# --------------------------------------------------
filtered_df["GK Shot Stopping"] = (
    filtered_df["xG Allowed"] - filtered_df["Goals Allowed"]
)

filtered_df["Defensive Actions"] = (
    filtered_df["Tackles"] + filtered_df["Interceptions"]
)

# --------------------------------------------------
# Section 1 — League overview
# --------------------------------------------------
st.subheader("📊 League Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Avg Goals Allowed",
    round(filtered_df["Goals Allowed"].mean(), 2)
)

col2.metric(
    "Avg xG Allowed",
    round(filtered_df["xG Allowed"].mean(), 2)
)

col3.metric(
    "Avg Save %",
    f"{round(filtered_df['Save%'].mean(), 1)}%"
)

# --------------------------------------------------
# Section 2 — Shot-stopping (GK performance)
# --------------------------------------------------
st.subheader("🧤 Goalkeeper Shot-Stopping")

fig_shotstop = px.scatter(
    filtered_df,
    x="xG Allowed",
    y="Goals Allowed",
    hover_name="team",
    size="Shots on Target Allowed",
    color="GK Shot Stopping",
    color_continuous_scale="RdBu",
    title="Goals Allowed vs xG Allowed",
    labels={
        "xG Allowed": "Expected Goals Against",
        "Goals Allowed": "Goals Conceded"
    }
)

fig_shotstop.add_shape(
    type="line",
    x0=filtered_df["xG Allowed"].min(),
    y0=filtered_df["xG Allowed"].min(),
    x1=filtered_df["xG Allowed"].max(),
    y1=filtered_df["xG Allowed"].max(),
    line=dict(dash="dash")
)

st.plotly_chart(fig_shotstop, use_container_width=True)

st.caption(
    "Above the diagonal = conceding more than expected (poor shot-stopping). "
    "Below = strong goalkeeping."
)

# --------------------------------------------------
# Section 3 — Save efficiency
# --------------------------------------------------
st.subheader("🧮 Save Efficiency")

fig_save = px.bar(
    filtered_df.sort_values("Save%", ascending=False),
    x="team",
    y="Save%",
    title="Save Percentage by Team"
)

st.plotly_chart(fig_save, use_container_width=True)

# --------------------------------------------------
# Section 4 — Defensive activity
# --------------------------------------------------
st.subheader("🧱 Defensive Activity")

fig_defense = px.scatter(
    filtered_df,
    x="Defensive Actions",
    y="Goals Allowed",
    hover_name="team",
    size="Clearances",
    color="Blocks",
    title="Defensive Actions vs Goals Allowed",
    labels={
        "Defensive Actions": "Tackles + Interceptions",
        "Goals Allowed": "Goals Conceded"
    }
)

st.plotly_chart(fig_defense, use_container_width=True)

# --------------------------------------------------
# Section 5 — Errors & discipline
# --------------------------------------------------
st.subheader("⚠️ Errors & Discipline")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Avg Errors",
    round(filtered_df["Errors"].mean(), 2)
)

col2.metric(
    "Avg Fouls",
    round(filtered_df["Fouls"].mean(), 1)
)

col3.metric(
    "Avg Yellow Cards",
    round(filtered_df["Yellow Cards"].mean(), 1)
)

# --------------------------------------------------
# Section 6 — Team table
# --------------------------------------------------
st.subheader("📋 Defensive Summary Table")

table_cols = [
    "team",
    "Goals Allowed",
    "xG Allowed",
    "GK Shot Stopping",
    "Save%",
    "Clean Sheet%",
    "Defensive Actions",
    "Blocks",
    "Clearances",
    "Errors"
]

st.dataframe(
    filtered_df[table_cols]
    .sort_values("Goals Allowed"),
    use_container_width=True
)
