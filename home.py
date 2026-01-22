# Home page for the Streamlit Soccer Data Analytics App

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
st.title("European Top 5 Leagues - Team Performance Analytics")
st.markdown(
    "Interactive analysis of **attacking**, **defensive**, and **possession** metrics "
    "across Europe's top five leagues."
)

st.divider()

# --------------------
# Sidebar filters
# --------------------
st.sidebar.header("Filters")

league = st.sidebar.multiselect(
    "Select League(s)",
    options=sorted(df["league"].unique()),
    default=sorted(df["league"].unique())
)

season = st.sidebar.multiselect(
    "Select Season(s)",
    options=sorted(df["season"].unique()),
    default=[df["season"].max()]
)

filtered_df = df[
    (df["league"].isin(league)) &
    (df["season"].isin(season))
]

# --------------------
# KPI metrics
# --------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Avg Goals / Match", round(filtered_df["Goals"].mean(), 2))

with col2:
    st.metric("Avg xG", round(filtered_df["xG"].mean(), 2))

with col3:
    st.metric("Avg Possession %", round(filtered_df["Poss"].mean(), 1))

with col4:
    st.metric("Avg Goal Difference", round(filtered_df["Goal Difference"].mean(), 2))

with col5:
    st.metric("Clean Sheet %", round(filtered_df["Clean Sheet%"].mean(), 1))

st.divider()

# --------------------
# League comparison charts
# --------------------
st.subheader("League Comparison")

league_summary = (
    filtered_df
    .groupby("league")
    .agg({
        "Goals": "mean",
        "xG": "mean",
        "Poss": "mean"
    })
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:
    fig_goals = px.bar(
        league_summary,
        x="league",
        y="Goals",
        title="Average Goals per Team",
        labels={"Goals": "Goals"}
    )
    st.plotly_chart(fig_goals, use_container_width=True)

with col2:
    fig_xg = px.bar(
        league_summary,
        x="league",
        y="xG",
        title="Average Expected Goals (xG)",
        labels={"xG": "xG"}
    )
    st.plotly_chart(fig_xg, use_container_width=True)

st.divider()

# --------------------
# Goals vs xG scatter
# --------------------
st.subheader("Goals vs Expected Goals")

fig_scatter = px.scatter(
    filtered_df,
    x="xG",
    y="Goals",
    color="league",
    hover_name="team",
    title="Goals vs xG (Over / Underperformance)",
    labels={"xG": "Expected Goals", "Goals": "Goals Scored"}
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# --------------------
# Top teams table
# --------------------
st.subheader("Top Teams Snapshot")

top_teams = (
    filtered_df
    .sort_values("Pts pMatch", ascending=False)
    .loc[:, [
        "team",
        "league",
        "Pts pMatch",
        "Goals",
        "xG",
        "Goal Difference",
        "Poss"
    ]]
    .head(10)
)

st.dataframe(
    top_teams,
    use_container_width=True,
    hide_index=True
)

# --------------------
# Footer hint
# --------------------
st.info(
    "Use the sidebar to filter leagues and seasons, or explore deeper analysis "
    "from the navigation menu."
)
