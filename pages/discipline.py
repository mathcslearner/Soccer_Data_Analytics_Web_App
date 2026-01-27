#Discipline page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Team Discipline", layout="wide")
st.title("Team Discipline Analysis")
st.markdown(
    """
    Analyze how disciplined teams are across leagues and seasons.
    Includes fouls, cards, penalties conceded, and errors — both raw and normalized.
    """
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("team_stats_2024-2025.csv")
    return df

df = load_data()

# --------------------------------------------------
# Filters
# --------------------------------------------------
with st.sidebar:
    st.header("Filters")
    league = st.multiselect(
        "League",
        sorted(df["league"].unique()),
        default=sorted(df["league"].unique())
    )
    season = st.multiselect(
        "Season",
        sorted(df["season"].unique()),
        default=sorted(df["season"].unique())
    )

df = df[
    (df["league"].isin(league)) &
    (df["season"].isin(season))
]

# --------------------------------------------------
# Derived discipline metrics
# --------------------------------------------------
df["Minutes"] = df["W"] * 90 + df["D"] * 90 + df["L"] * 90

df["Fouls_p90"] = df["Fouls"] / df["Minutes"] * 90
df["Yellow_p90"] = df["Yellow Cards"] / df["Minutes"] * 90
df["Red_p90"] = df["Red Cards"] / df["Minutes"] * 90
df["PensConceded_p90"] = df["Pen Conceded"] / df["Minutes"] * 90
df["Errors_p90"] = df["Errors"] / df["Minutes"] * 90

# Possession-adjusted fouls
df["Fouls_adjPoss"] = df["Fouls_p90"] / (1 - df["Poss"] / 100)

# Composite Discipline Index (lower = better)
df["Discipline_Index"] = (
    df["Yellow_p90"] * 0.4 +
    df["Red_p90"] * 1.5 +
    df["Fouls_p90"] * 0.3 +
    df["PensConceded_p90"] * 0.6 +
    df["Errors_p90"] * 0.4
)

# --------------------------------------------------
# League-level overview
# --------------------------------------------------
st.subheader("📊 Average Discipline Metrics by League")

metrics = {
    "Fouls per 90": "Fouls_p90",
    "Yellow Cards per 90": "Yellow_p90",
    "Red Cards per 90": "Red_p90",
    "Penalties Conceded per 90": "PensConceded_p90"
}

league_avg = (
    df.groupby("league")[list(metrics.values())]
    .mean()
    .reset_index()
)

league_avg = league_avg.sort_values("Fouls_p90")

cols = st.columns(2)

for i, (label, metric) in enumerate(metrics.items()):
    fig = px.bar(
        league_avg,
        x="league",
        y=metric,
        title=label
    )

    fig.update_traces(
        texttemplate="%{y:.2f}",
        textposition="outside"
    )

    y_max = league_avg[metric].max()
    fig.update_yaxes(range=[0, y_max * 1.15])

    cols[i % 2].plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Aggression vs Discipline scatter
# --------------------------------------------------
st.subheader("⚖️ Aggression vs Control")

fig = px.scatter(
    df,
    x="Fouls_p90",
    y="Yellow_p90",
    size="Red_p90",
    color="league",
    hover_name="team",
    title="Fouls vs Yellow Cards (Bubble = Red Cards)"
)
fig.add_hline(y=df["Yellow_p90"].mean(), line_dash="dash")
fig.add_vline(x=df["Fouls_p90"].mean(), line_dash="dash")

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Most disciplined / least disciplined teams
# --------------------------------------------------
st.subheader("Team Rankings (Discipline Index)")

best = df.sort_values("Discipline_Index").head(10)
worst = df.sort_values("Discipline_Index", ascending=False).head(10)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Most Disciplined Teams")
    st.dataframe(
        best[[
            "team",
            "league",
            "Discipline_Index",
            "Fouls_p90",
            "Yellow_p90",
            "Red_p90"
        ]],
        hide_index=True
    )

with col2:
    st.markdown("### Least Disciplined Teams")
    st.dataframe(
        worst[[
            "team",
            "league",
            "Discipline_Index",
            "Fouls_p90",
            "Yellow_p90",
            "Red_p90"
        ]],
        hide_index=True
    )

# --------------------------------------------------
# Team deep dive
# --------------------------------------------------
st.subheader("Team Discipline Profile")

team = st.selectbox("Select Team", sorted(df["team"].unique()))
team_df = df[df["team"] == team]

radar_metrics = [
    "Fouls_p90",
    "Yellow_p90",
    "Red_p90",
    "PensConceded_p90",
    "Errors_p90"
]

radar_vals = team_df[radar_metrics].iloc[0]
league_avg_vals = df[radar_metrics].mean()

radar_df = pd.DataFrame({
    "Metric": radar_metrics,
    "Team": radar_vals.values,
    "League Avg": league_avg_vals.values
})

fig = px.line_polar(
    radar_df,
    r="Team",
    theta="Metric",
    line_close=True,
    title=f"{team} – Discipline Radar"
)
fig.add_trace(
    px.line_polar(
        radar_df,
        r="League Avg",
        theta="Metric",
        line_close=True
    ).data[0]
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Interpretation helper
# --------------------------------------------------
st.info(
    """
    **How to read this page**
    - High fouls + low cards → aggressive but controlled
    - High cards + penalties → reckless defending
    - Low Discipline Index → cleaner, more controlled teams
    """
)
