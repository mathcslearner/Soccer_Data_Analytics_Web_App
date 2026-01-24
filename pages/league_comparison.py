# League comparison page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
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

# --------
# Page
# -----------

def league_comparison_page(df_all):
    st.title("League Comparison")
    st.write("Compare how stats differ across leagues in a season.")

    # ---- selectors
    season = st.selectbox("Season", sorted(df_all["season"].unique()))

    # Mean vs Median toggle
    summary_choice = st.radio("League summary type", ["Mean", "Median"], horizontal=True)

    # Major stats to compare
    major_stats = [
        "Pts pMatch", "Goals", "xG", "Goal Difference", "Poss",
        "Shots", "SoT%", "Goals Allowed", "xG Allowed"
    ]

    # Filter season
    season_df = df_all[df_all["season"] == season]

    # ---- league summary table (mean/median)
    league_summary = (
        season_df
        .groupby("league")[major_stats]
        .agg("mean" if summary_choice == "Mean" else "median")
        .reset_index()
    )

    # Color best league for each stat
    def highlight_best(row):
        best = row.max()
        return ['background-color: lightgreen' if v == best else '' for v in row]

    st.subheader("League Averages")
    st.dataframe(
        league_summary.style
            .apply(highlight_best, subset=major_stats, axis=0)
            .format({col: "{:.1f}" for col in major_stats}),
        use_container_width=True
    )

    # ---- custom soccer-themed colors
    league_color_map = {
        "EPL": "#e10600",        # Red
        "La Liga": "#1d4ed8",    # Blue
        "Serie A": "#16a34a",    # Green
        "Bundesliga": "#f59e0b", # Yellow
        "Ligue 1": "#7c3aed"     # Purple
    }

    # ---- grouped bar charts for all major stats
    st.subheader("Grouped Bar Charts (League Averages)")
    bar_rows = 3  # how many charts per row

    for i in range(0, len(major_stats), bar_rows):
        cols = st.columns(bar_rows)
        for j, stat in enumerate(major_stats[i:i + bar_rows]):
            bar_df = league_summary[["league", stat]]
            fig_bar = px.bar(
                bar_df,
                x="league",
                y=stat,
                title=stat,
                text=stat,
                color="league",
                color_discrete_map=league_color_map
            )
            with cols[j]:
                st.plotly_chart(fig_bar, use_container_width=True)

    # ---- boxplots for distribution differences
    st.subheader("Distribution Comparison (Boxplots)")

    # Show boxplots in grid layout
    box_rows = 3  # how many charts per row
    for i in range(0, len(major_stats), box_rows):
        cols = st.columns(box_rows)
        for j, stat in enumerate(major_stats[i:i + box_rows]):
            box_df = season_df[["league", stat]].copy()
            box_df[stat] = box_df[stat].round(1)

            fig_box = px.box(
                box_df,
                x="league",
                y=stat,
                title=stat,
                points="outliers",
                color="league",
                color_discrete_map=league_color_map
            )
            with cols[j]:
                st.plotly_chart(fig_box, use_container_width=True)

    # ---- violin plots for distribution differences
    st.subheader("Distribution Comparison (Violin Plots)")

    violin_rows = 3  # how many charts per row
    for i in range(0, len(major_stats), violin_rows):
        cols = st.columns(violin_rows)
        for j, stat in enumerate(major_stats[i:i + violin_rows]):
            violin_df = season_df[["league", stat]].copy()
            violin_df[stat] = violin_df[stat].round(1)

            fig_violin = px.violin(
                violin_df,
                x="league",
                y=stat,
                title=stat,
                box=True,           # show inner boxplot
                points="outliers",  # match boxplot behavior
                color="league",
                color_discrete_map=league_color_map
            )

            fig_violin.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=40, r=40, t=50, b=40)
            )

            with cols[j]:
                st.plotly_chart(fig_violin, use_container_width=True)

league_comparison_page(df)