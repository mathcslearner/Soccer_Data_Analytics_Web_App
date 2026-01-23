#Team Analysis Page

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Team Analysis", layout="wide")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("team_stats_2024-2025.csv")

df = load_data()

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def league_average(df, league, season, metric):
    return df[
        (df["league"] == league) &
        (df["season"] == season)
    ][metric].mean()

def league_rank(df, league, season, team, metric, ascending=False):
    league_df = (
        df[
            (df["league"] == league) &
            (df["season"] == season)
        ][["team", metric]]
        .sort_values(metric, ascending=ascending)
        .reset_index(drop=True)
    )

    league_df["rank"] = league_df.index + 1
    return league_df.loc[league_df["team"] == team, "rank"].values[0]

def league_percentile(df, league, season, team, metric, ascending=False):
    league_df = df[
        (df["league"] == league) &
        (df["season"] == season)
    ][["team", metric]]

    league_df["rank"] = league_df[metric].rank(
        ascending=ascending,
        method="min"
    )

    percentile = 100 * (1 - (league_df.loc[league_df["team"] == team, "rank"].values[0] - 1) / (len(league_df) - 1))
    return percentile

def league_mean_std(df, league, season, metric):
    league_df = df[
        (df["league"] == league) &
        (df["season"] == season)
    ][metric]

    return league_df.mean(), league_df.std()

def league_min_max(df, league, season, metric):
    league_df = df[
        (df["league"] == league) &
        (df["season"] == season)
    ][metric]

    return league_df.min(), league_df.max()

def metric_card(df, league, season, team, metric, label, asc=False, decimals=1):
    value = df[metric].values[0]
    avg = league_average(df_all, league, season, metric)
    rank = league_rank(df_all, league, season, team, metric, ascending=asc)

    st.metric(
        label,
        f"{value:.{decimals}f}"
    )
    st.write(f"Avg: {avg:.{decimals}f} | Rank: {rank}")

# Radar builder
def get_radar_values(df, league, season, team, metrics, radar_mode):
    values = []

    team_row = df[
        (df["league"] == league) &
        (df["season"] == season) &
        (df["team"] == team)
    ].iloc[0]

    for metric, _, ascending in metrics:

        if radar_mode == "Percentiles":
            val = league_percentile(
                df, league, season, team, metric, ascending
            )

        elif radar_mode == "Min-max scaled":
            min_val, max_val = league_min_max(df, league, season, metric)

            if max_val == min_val:
                val = 50
            else:
                if ascending:
                    # lower is better (e.g. Goals Allowed)
                    val = 100 * (max_val - team_row[metric]) / (max_val - min_val)
                else:
                    # higher is better
                    val = 100 * (team_row[metric] - min_val) / (max_val - min_val)

            # safety clamp
            val = max(0, min(100, val))

        else:  # Normalized (z-score)
            mean, std = league_mean_std(df, league, season, metric)

            if std == 0:
                val = 0
            else:
                val = (team_row[metric] - mean) / std

            if ascending:
                val = -val

        values.append(val)

    return values

def build_radar(df, league, season, team, metrics, title, radar_mode):
    categories = [label for _, label, _ in metrics]

    values = get_radar_values(
        df, league, season, team, metrics, radar_mode
    )

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name=team
    ))

    # Reference line
    if radar_mode in ["Percentiles", "Min-max scaled"]:
        ref_values = [50] * len(categories)
        radial_range = [0, 100]
        ref_name = "League Avg"

    elif radar_mode == "Normalized (z-score)":
        ref_values = [0] * len(categories)
        radial_range = [-3, 3]  # standard & interpretable
        ref_name = "League Avg (0)"

    else:
        ref_values = None
        radial_range = None

    if ref_values:
        fig.add_trace(go.Scatterpolar(
            r=ref_values,
            theta=categories,
            fill="toself",
            opacity=0.25,
            name=ref_name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=radial_range
            )
        ),
        title=title,
        showlegend=True
    )

    return fig

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("Team Analysis")
st.markdown("League-relative performance breakdown for an individual team.")

st.divider()

# --------------------------------------------------
# Filters
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    league = st.selectbox("League", sorted(df["league"].unique()))

with col2:
    season = st.selectbox(
        "Season",
        sorted(df[df["league"] == league]["season"].unique())
    )

with col3:
    team = st.selectbox(
        "Team",
        sorted(
            df[
                (df["league"] == league) &
                (df["season"] == season)
            ]["team"].unique()
        )
    )

df_all = df.copy()

team_df = df[
    (df["league"] == league) &
    (df["season"] == season) &
    (df["team"] == team)
]

st.divider()

# --------------------------------------------------
# Key Metrics
# --------------------------------------------------
st.subheader(f"{team} — Key Metrics (League Context)")

key_metrics = [
    ("Pts pMatch", "Pts / Match", False, 2),
    ("Goals", "Goals", False, 0),
    ("xG", "xG", False, 1),
    ("Goal Difference", "Goal Difference", False, 0),
    ("Poss", "Possession %", False, 1),
]

cols = st.columns(len(key_metrics))

for col, (metric, label, asc, dec) in zip(cols, key_metrics):
    with col:
        metric_card(team_df, league, season, team, metric, label, asc, dec)

st.caption("Ranks computed within selected league and season.")

st.divider()

# --------------------------------------------------
# Attacking & Build-up Profile
# --------------------------------------------------
st.subheader("Attacking & Build-up Profile")

attack_metrics = [
    ("Shots", "Shots", False, 0),
    ("Shots on Target", "Shots on Target", False, 0),
    ("SoT%", "SoT %", False, 1),
    ("Goals per Shots", "Goals / Shot", False, 2),
    ("Pass Completion%", "Pass Completion %", False, 1),
    ("Prg Passes", "Progressive Passes", False, 0),
    ("Prg Carries", "Progressive Carries", False, 0),
]

cols = st.columns(4)

for i, (metric, label, asc, dec) in enumerate(attack_metrics):
    with cols[i % 4]:
        metric_card(team_df, league, season, team, metric, label, asc, dec)

st.divider()

# --------------------------------------------------
# Defensive Profile
# --------------------------------------------------
st.subheader("Defensive Profile")

def_metrics = [
    ("Goals Allowed", "Goals Allowed", True, 0),
    ("xG Allowed", "xG Allowed", True, 1),
    ("Tackle+Interceptions", "Tackles + Interceptions", False, 0),
    ("Clean Sheet", "Clean Sheets", False, 0),
]

cols = st.columns(4)

for col, (metric, label, asc, dec) in zip(cols, def_metrics):
    with col:
        metric_card(team_df, league, season, team, metric, label, asc, dec)

# --------------
# Radar Section
# -------------
        
radar_mode = st.radio(
    "Radar scale",
    ["Percentiles", "Min-max scaled", "Normalized (z-score)"],
    horizontal=True
)


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

st.subheader("Overall Team Profile (League Percentiles)")

fig_overall = build_radar(
    df_all, league, season, team,
    overall_metrics,
    f"{team} — Overall Profile",
    radar_mode
)

st.plotly_chart(fig_overall, use_container_width=True)

st.subheader("Offensive vs Defensive Profiles")

col1, col2 = st.columns(2)

with col1:
    fig_attack = build_radar(
        df_all, league, season, team,
        offensive_metrics,
        f"{team} — Offensive Profile",
        radar_mode
    )
    st.plotly_chart(fig_attack, use_container_width=True)

with col2:
    fig_def = build_radar(
        df_all, league, season, team,
        defensive_metrics,
        f"{team} — Defensive Profile",
        radar_mode
    )
    st.plotly_chart(fig_def, use_container_width=True)

