#Team Analysis Page

import streamlit as st
import pandas as pd

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


def metric_card(df, league, season, team, metric, label, asc=False, decimals=1):
    value = df[metric].values[0]
    avg = league_average(df_all, league, season, metric)
    rank = league_rank(df_all, league, season, team, metric, ascending=asc)

    st.metric(
        label,
        f"{value:.{decimals}f}"
    )
    st.write(f"Avg: {avg:.{decimals}f} | Rank: {rank}")


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

