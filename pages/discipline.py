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
st.subheader("Average Discipline Metrics by League")

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
st.subheader("Aggression vs Control")

x_mean = df["Fouls_p90"].mean()
y_mean = df["Yellow_p90"].mean()

fig = px.scatter(
    df,
    x="Fouls_p90",
    y="Yellow_p90",
    size="Red_p90",
    color="league",
    hover_name="team",
    title="Aggression vs Control Map"
)

# Bold quadrant axes
fig.add_vline(
    x=x_mean,
    line_width=2,
    line_color="white",
    line_dash="dash"
)

fig.add_hline(
    y=y_mean,
    line_width=2,
    line_color="white",
    line_dash="dash"
)

# Force vertical stretch
y_span = df["Yellow_p90"].max() - df["Yellow_p90"].min()
if y_span == 0:
    y_span = y_mean * 0.5

fig.update_yaxes(
    autorange=False,
    range=[y_mean - 0.8 * y_span, y_mean + 0.8 * y_span]
)

# Quadrant annotations
fig.add_annotation(x=x_mean * 0.7, y=y_mean * 1.4,
                   text="Passive but Reckless",
                   showarrow=False)

fig.add_annotation(x=x_mean * 1.3, y=y_mean * 1.4,
                   text="Aggressive & Reckless",
                   showarrow=False)

fig.add_annotation(x=x_mean * 0.7, y=y_mean * 0.6,
                   text="Passive & Controlled",
                   showarrow=False)

fig.add_annotation(x=x_mean * 1.3, y=y_mean * 0.6,
                   text="Aggressive but Controlled",
                   showarrow=False)

fig.update_layout(height=800)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    """
    **Quadrant interpretation**
    - Aggressive & Controlled: press hard, foul often, but avoid cards
    - Aggressive & Reckless: foul-heavy with frequent bookings
    - Passive & Controlled: low defensive intensity, clean play
    - Passive but Reckless: fewer fouls, but poor timing or positioning
    """
)

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
st.subheader("🔍 Team Discipline Profile")

team = st.selectbox("Select Team", sorted(df["team"].unique()))
team_df = df[df["team"] == team]

radar_metrics = [
    "Fouls_p90",
    "Yellow_p90",
    "Red_p90",
    "PensConceded_p90",
    "Errors_p90"
]

# Percentile normalization
radar_norm = df[radar_metrics].rank(pct=True)

team_vals = radar_norm.loc[team_df.index[0]]
league_avg_vals = radar_norm.mean()

radar_plot_df = pd.DataFrame({
    "Metric": radar_metrics * 2,
    "Value": list(team_vals.values) + list(league_avg_vals.values),
    "Group": ["Team"] * len(radar_metrics) + ["League Avg"] * len(radar_metrics)
})

fig = px.line_polar(
    radar_plot_df,
    r="Value",
    theta="Metric",
    color="Group",
    line_close=True,
    title=f"{team} – Discipline Radar (Percentile Normalized)"
)

fig.update_traces(fill="toself")

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            range=[0, 1],
            showticklabels=True,
            ticks="outside"
        )
    )
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
