# League comparison page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import numpy as np

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

# ---------------------
# Helper functions
# -------------------------

def compute_hull(points):
    if len(points) < 3:
        return None
    hull = ConvexHull(points)
    return points[hull.vertices]

def compute_ellipse(x, y, n_std=1.5, n_points=100):
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    theta = np.linspace(0, 2 * np.pi, n_points)
    ellipse = np.array([
        n_std * np.sqrt(eigvals[0]) * np.cos(theta),
        n_std * np.sqrt(eigvals[1]) * np.sin(theta)
    ])

    ellipse_rot = eigvecs @ ellipse

    return (
        ellipse_rot[0] + mean_x,
        ellipse_rot[1] + mean_y
    )

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
        "ENG-Premier League": "#e10600",        # Red
        "ESP-La Liga": "#1d4ed8",    # Blue
        "ITA-Serie A": "#16a34a",    # Green
        "GER-Bundesliga": "#f59e0b", # Yellow
        "FRA-Ligue 1": "#7c3aed"     # Purple
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
    
    # ---- attack vs defense 2d plot
    attack_metrics = [
        "Goals",
        "xG",
        "Shots",
        "Poss"
    ]

    defense_metrics = [
        "Goals Allowed",
        "xG Allowed",
        "Shots on Target Allowed"
    ]

    # Build attack/defense index per league
    def build_attack_defense_index(df):
        league_rows = []

        for league in sorted(df["league"].unique()):
            league_df = df[df["league"] == league]

            attack_score = league_df[attack_metrics].mean().mean()

            defense_score = (
                -league_df[defense_metrics].mean()  # invert defensive metrics
            ).mean()

            league_rows.append({
                "League": league,
                "Attack Index": attack_score,
                "Defense Index": defense_score
            })

        idx_df = pd.DataFrame(league_rows)

        # Normalize (z-score)
        idx_df["Attack Index"] = (
            (idx_df["Attack Index"] - idx_df["Attack Index"].mean())
            / idx_df["Attack Index"].std()
        )

        idx_df["Defense Index"] = (
            (idx_df["Defense Index"] - idx_df["Defense Index"].mean())
            / idx_df["Defense Index"].std()
        )

        return idx_df.round(2)

    # Plot leagues in 2d space
    def attack_defense_scatter(idx_df, league_colors):
        fig = go.Figure()

        # League points
        for _, row in idx_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Attack Index"]],
                y=[row["Defense Index"]],
                mode="markers+text",
                text=[row["League"]],
                textposition="top center",
                marker=dict(
                    size=18,
                    color=league_colors.get(row["League"]),
                    line=dict(width=1, color="black")
                ),
                name=row["League"]
            ))

        # --- symmetric axis limits
        max_attack = abs(idx_df["Attack Index"]).max()
        max_defense = abs(idx_df["Defense Index"]).max()
        axis_limit = max(max_attack, max_defense) * 1.1  # small padding

        # Reference lines
        fig.add_hline(
            y=0,
            line_width=2,
            line_color="rgba(120,120,120,0.9)"
        )

        fig.add_vline(
            x=0,
            line_width=2,
            line_color="rgba(120,120,120,0.9)"
        )

        fig.update_layout(
            title="League Attack vs Defense Index",
            xaxis=dict(
                title="Attack Strength (↑ better)",
                range=[-axis_limit, axis_limit],
                zeroline=False
            ),
            yaxis=dict(
                title="Defensive Strength (↑ better)",
                range=[-axis_limit, axis_limit],
                zeroline=False
            ),
            height=480,
            showlegend=False,
            margin=dict(l=60, r=40, t=60, b=50)
        )

        return fig

    st.subheader("Attack vs Defense Landscape")

    idx_df = build_attack_defense_index(df_all)

    st.plotly_chart(
        attack_defense_scatter(idx_df, league_color_map),
        use_container_width=True
    )

    # Plot teams in 2d space
    def build_team_attack_defense(df):
        rows = []

        for _, row in df.iterrows():
            attack = row[attack_metrics].mean()
            defense = -row[defense_metrics].mean()  # invert defense

            rows.append({
                "league": row["league"],
                "team": row["team"],
                "Attack Index": attack,
                "Defense Index": defense
            })

        idx_df = pd.DataFrame(rows)

        # Normalize globally (important for fair hull comparison)
        idx_df["Attack Index"] = (
            (idx_df["Attack Index"] - idx_df["Attack Index"].mean())
            / idx_df["Attack Index"].std()
        )

        idx_df["Defense Index"] = (
            (idx_df["Defense Index"] - idx_df["Defense Index"].mean())
            / idx_df["Defense Index"].std()
        )

        return idx_df

    def single_league_shape_plot(
        league_df,
        league_name,
        league_color,
        shape_mode,
        axis_limit
    ):
        fig = go.Figure()

        x = league_df["Attack Index"].values
        y = league_df["Defense Index"].values
        points = league_df[["Attack Index", "Defense Index"]].values

        # ---- team dots
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=8,
                color=league_color,
                opacity=0.6
            ),
            text=league_df["team"],
            hovertemplate="<b>%{text}</b><br>Attack: %{x:.2f}<br>Defense: %{y:.2f}<extra></extra>"
        ))

        # ---- centroid dot
        centroid_x = league_df["Attack Index"].mean()
        centroid_y = league_df["Defense Index"].mean()

        fig.add_trace(go.Scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode="markers+text",
            marker=dict(
                size=16,
                color="black",
                symbol="diamond"
            ),
            text=["Centroid"],
            textposition="top center",
            hovertemplate="<b>League Centroid</b><br>Attack: %{x:.2f}<br>Defense: %{y:.2f}<extra></extra>"
        ))

        # ---- hull or ellipse
        if shape_mode == "Convex Hull":
            hull_pts = compute_hull(points)
            if hull_pts is not None:
                hx = np.append(hull_pts[:, 0], hull_pts[0, 0])
                hy = np.append(hull_pts[:, 1], hull_pts[0, 1])

                fig.add_trace(go.Scatter(
                    x=hx,
                    y=hy,
                    fill="toself",
                    fillcolor=league_color,
                    line=dict(color=league_color, width=2),
                    opacity=0.25
                ))
        else:
            ex, ey = compute_ellipse(x, y)
            fig.add_trace(go.Scatter(
                x=ex,
                y=ey,
                fill="toself",
                fillcolor=league_color,
                line=dict(color=league_color, width=2),
                opacity=0.3
            ))

        # ---- reference lines
        fig.add_hline(y=0, line_width=2, line_color="rgba(120,120,120,0.9)")
        fig.add_vline(x=0, line_width=2, line_color="rgba(120,120,120,0.9)")

        fig.update_layout(
            title=league_name,
            xaxis=dict(
                title="Attack Strength",
                range=[-axis_limit, axis_limit],
                zeroline=False
            ),
            yaxis=dict(
                title="Defensive Strength",
                range=[-axis_limit, axis_limit],
                zeroline=False
            ),
            height=350,
            margin=dict(l=50, r=20, t=50, b=40),
            showlegend=False
        )

        return fig

    
    season_df = df_all[df_all["season"] == season]

    team_idx_df = build_team_attack_defense(season_df)

    max_val = max(
        abs(team_idx_df["Attack Index"]).max(),
        abs(team_idx_df["Defense Index"]).max()
    ) * 1.1

    season_df = df_all[df_all["season"] == season]

    st.subheader("League Style Diversity (Attack vs Defense)")

    shape_mode = st.radio(
        "Display mode",
        ["Convex Hull", "Ellipse (typical range)"],
        horizontal=True
    )

    leagues = sorted(team_idx_df["league"].unique())

    row1 = st.columns(3)
    row2 = st.columns(3)

    # First row: 3 leagues
    for col, league in zip(row1, leagues[:3]):
        league_df = team_idx_df[team_idx_df["league"] == league]
        with col:
            st.plotly_chart(
                single_league_shape_plot(
                    league_df,
                    league,
                    league_color_map[league],
                    shape_mode,
                    max_val
                ),
                use_container_width=True
            )

    # Second row: 2 leagues + 1 empty column
    for col, league in zip(row2[:2], leagues[3:]):
        league_df = team_idx_df[team_idx_df["league"] == league]
        with col:
            st.plotly_chart(
                single_league_shape_plot(
                    league_df,
                    league,
                    league_color_map[league],
                    shape_mode,
                    max_val
                ),
                use_container_width=True
            )

league_comparison_page(df)