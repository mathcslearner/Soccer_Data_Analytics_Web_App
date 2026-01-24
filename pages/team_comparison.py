# Team Comparison page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Team Comparison", layout="wide")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("team_stats_2024-2025.csv")

df = load_data()

# ------------------------------------------------------------
# Helper functions 
# ------------------------------------------------------------
def league_percentile(df, league, season, team, metric, ascending=False):
    league_df = df[(df["league"] == league) & (df["season"] == season)][["team", metric]]
    league_df["rank"] = league_df[metric].rank(ascending=ascending, method="min")
    percentile = 100 * (1 - (league_df.loc[league_df["team"] == team, "rank"].values[0] - 1) / (len(league_df) - 1))
    return percentile

def league_mean_std(df, league, season, metric):
    league_df = df[(df["league"] == league) & (df["season"] == season)][metric]
    return league_df.mean(), league_df.std()

def league_distribution(df, league, season, metric):
    league_df = df[(df["league"] == league) & (df["season"] == season)][metric]
    return {
        "min": league_df.min(),
        "max": league_df.max(),
        "mean": league_df.mean(),
        "median": league_df.median()
    }

def get_radar_values(df, league, season, team, metrics, radar_mode, center_reference=None):
    values = []
    team_row = df[(df["league"] == league) & (df["season"] == season) & (df["team"] == team)].iloc[0]

    for metric, _, ascending in metrics:
        if radar_mode == "Percentiles":
            val = league_percentile(df, league, season, team, metric, ascending)

        elif radar_mode == "Min-Max scaled":
            stats = league_distribution(df, league, season, metric)
            min_val, max_val = stats["min"], stats["max"]
            center = stats["mean"] if center_reference == "League mean" else stats["median"]

            if max_val - min_val < 1e-6:
                val = 50
            else:
                if team_row[metric] >= center:
                    val = 50 + 50 * (team_row[metric] - center) / (max_val - center)
                else:
                    val = 50 * (team_row[metric] - min_val) / (center - min_val)

            if ascending:
                val = 100 - val

        else:  # Z-score
            mean, std = league_mean_std(df, league, season, metric)
            val = 0 if std == 0 else (team_row[metric] - mean) / std
            if ascending:
                val = -val

        values.append(val)

    return values

def build_radar(df, league, season, team, metrics, title, radar_mode, center_reference=None):
    categories = [label for _, label, _ in metrics]
    values = get_radar_values(df, league, season, team, metrics, radar_mode, center_reference)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself", name=team))

    if radar_mode in ["Percentiles", "Min-Max scaled"]:
        fig.add_trace(
            go.Scatterpolar(
                r=[50]*len(categories),
                theta=categories,
                fill="toself",
                opacity=0.25,
                name="League Avg"
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100] if radar_mode != "Z-score" else None)),
        title=title,
        showlegend=True
    )
    return fig

def highlight_better(row, team1, team2, lower_better_map):
    styles = [""] * len(row)

    v1 = row[team1]
    v2 = row[team2]
    lower_better = lower_better_map[row.name]

    if pd.isna(v1) or pd.isna(v2) or v1 == v2:
        return styles

    # Determine winner
    if lower_better:
        winner = team1 if v1 < v2 else team2
        diff = abs(v1 - v2)
        denom = abs(min(v1, v2))
    else:
        winner = team1 if v1 > v2 else team2
        diff = abs(v1 - v2)
        denom = abs(max(v1, v2))

    # Safe percentage difference
    pct_diff = 0 if denom < 1e-6 else diff / denom

    # Map % diff → opacity (subtle → strong)
    intensity = min(0.85, 0.15 + pct_diff)

    styles[row.index.get_loc(winner)] = (
        f"background-color: rgba(30, 144, 255, {intensity})"
    )

    return styles

def percent_diff(v1, v2, lower_better):
    if pd.isna(v1) or pd.isna(v2) or v1 == v2:
        return 0.0

    if lower_better:
        better = min(v1, v2)
        worse = max(v1, v2)
    else:
        better = max(v1, v2)
        worse = min(v1, v2)

    denom = abs(better)
    return 0.0 if denom < 1e-6 else 100 * abs(worse - better) / denom

# ------------------------------------------------------------
# Metric sets 
# ------------------------------------------------------------
overall_metrics = [
    ("Goals", "Goals", False),
    ("xG", "xG", False),
    ("Poss", "Possession", False),
    ("Prg Passes", "Prog Passes", False),
    ("Goals Allowed", "Goals Allowed", True),
    ("xG Allowed", "xG Allowed", True),
]

offensive_metrics = [
    ("Goals", "Goals", False),
    ("xG", "xG", False),
    ("Shots", "Shots", False),
    ("SoT%", "SoT %", False),
    ("Goals per Shots", "Goals / Shot", False),
    ("Pass Completion%", "Pass Comp %", False),
    ("Prg Passes", "Prog Passes", False),
    ("Prg Carries", "Prog Carries", False),
]

defensive_metrics = [
    ("Goals Allowed", "Goals Allowed", True),
    ("xG Allowed", "xG Allowed", True),
    ("Shots on Target Allowed", "SoT Allowed", True),
    ("Tackle+Interceptions", "Tackles + Int", False),
    ("Blocks", "Blocks", False),
    ("Clearances", "Clearances", False),
]

# ------------------------------------------------------------
# Page layout
# ------------------------------------------------------------
def team_comparison_page(df_all):
    st.title("Team Comparison")

    league = st.selectbox("League", sorted(df_all["league"].unique()))
    season = st.selectbox("Season", sorted(df_all["season"].unique()))

    teams = sorted(df_all[(df_all["league"] == league) & (df_all["season"] == season)]["team"].unique())
    team1 = st.selectbox("Team 1", teams, index=0)
    team2 = st.selectbox("Team 2", teams, index=1)

    radar_mode = st.radio(
        "Radar scale",
        ["Percentiles", "Min-Max scaled", "Z-score"],
        horizontal=True
    )

    center_reference = None
    if radar_mode == "Min-Max scaled":
        center_reference = st.radio(
            "Center scaling around",
            ["League mean", "League median"],
            horizontal=True
        )

    # ---------------- Radar charts ----------------
    st.subheader("Radar Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            build_radar(df_all, league, season, team1, overall_metrics, f"{team1} — Overall", radar_mode, center_reference),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            build_radar(df_all, league, season, team2, overall_metrics, f"{team2} — Overall", radar_mode, center_reference),
            use_container_width=True
        )

    # ---------------- Stat comparison table ----------------
    comparison_metrics = [
        ("Pts pMatch", False),
        ("Goals", False),
        ("xG", False),
        ("Goal Difference", False),
        ("Poss", False),
        ("Shots", False),
        ("SoT%", False),
        ("Goals Allowed", True),
        ("xG Allowed", True),
    ]

    st.subheader("Stat-by-Stat Comparison")

    row1 = df_all[(df_all["league"] == league) & (df_all["season"] == season) & (df_all["team"] == team1)].iloc[0]
    row2 = df_all[(df_all["league"] == league) & (df_all["season"] == season) & (df_all["team"] == team2)].iloc[0]

    rows = []
    for metric, lower_better in comparison_metrics:
        v1 = row1[metric]
        v2 = row2[metric]

        rows.append({
            "Metric": metric,
            team1: v1,
            team2: v2,
            "% Diff": percent_diff(v1, v2, lower_better),
            "lower_better": lower_better
        })

    comparison_df = pd.DataFrame(rows)

    lower_better_map = comparison_df["lower_better"].to_dict()

    styled_df = (
        comparison_df
        .drop(columns="lower_better")
        .style
        .apply(
            lambda row: highlight_better(row, team1, team2, lower_better_map),
            axis=1
        )
        .format({
            team1: "{:.2f}",
            team2: "{:.2f}",
            "% Diff": "{:.1f}%"
        })
    )

    st.dataframe(styled_df, use_container_width=True)

    # ---------------- Side-by-side bar chart ----------------
    st.subheader("Headline Stat Comparison")

    bar_metrics = ["Goals", "xG", "Shots", "Poss", "Goal Difference"]

    # First row: 3 charts
    row1_cols = st.columns(3)
    for i, metric in enumerate(bar_metrics[:3]):
        bar_df = pd.DataFrame({
            "Team": [team1, team2],
            "Value": [row1[metric], row2[metric]]
        })

        fig_bar = px.bar(
            bar_df,
            x="Team",
            y="Value",
            title=metric,
            text="Value"
        )

        with row1_cols[i]:
            st.plotly_chart(fig_bar, use_container_width=True)

    # Second row: 2 charts
    row2_cols = st.columns(2)
    for i, metric in enumerate(bar_metrics[3:]):
        bar_df = pd.DataFrame({
            "Team": [team1, team2],
            "Value": [row1[metric], row2[metric]]
        })

        fig_bar = px.bar(
            bar_df,
            x="Team",
            y="Value",
            title=metric,
            text="Value"
        )

        with row2_cols[i]:
            st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------------------------
# Run page
# ------------------------------------------------------------
team_comparison_page(df)

