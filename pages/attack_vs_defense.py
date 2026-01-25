#Attack vs Defense page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Tactical Identity", layout="wide")

st.title("⚔️ Tactical Identity Analysis")
st.markdown(
    """
    Analyze team style through Attack vs Defense, Build-up vs Attack, cluster radar, and tactical maps.
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
# Prepare metrics
# --------------------------------------------------
df = df.copy()
df["Opp_Poss"] = 100 - df["Poss"]

scaler = StandardScaler()

# Attack index
attack_features = ["xG", "Shots", "Touches Att 3rd", "SCA", "GCA"]
df[attack_features] = scaler.fit_transform(df[attack_features])
df["Attack_Index"] = df[attack_features].mean(axis=1)

# Pressing / defense behavior
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

# Defensive outcomes
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
df[defensive_outcome_features] = scaler.fit_transform(df[defensive_outcome_features])
df["Defensive_Solidity"] = df[defensive_outcome_features].mean(axis=1)

df["Defense_Index"] = 0.5 * df["Pressing_Index"] + 0.5 * df["Defensive_Solidity"]

# Build-up style index
build_features = [
    "Pass Completion%",
    "Passes Attempted",
    "Avg Pass Distance",
    "Prg Passes",
    "Progressive Receives",
    "Touches Att 3rd"
]
df[build_features] = scaler.fit_transform(df[build_features])
df["Build_Index"] = df[build_features].mean(axis=1)

# --------------------------------------------------
# Clustering
# --------------------------------------------------
st.sidebar.header("Clustering")
k = st.sidebar.slider("Number of playstyle clusters", 2, 8, 4)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df[["Attack_Index", "Defense_Index"]])

# Define cluster playstyle explanations
cluster_explanations = {
    0: "High press + strong attack (aggressive, dominant)",
    1: "Low block + efficient defense (compact, counter-based)",
    2: "Strong attack + weak defense (high risk, transition-heavy)",
    3: "Balanced / possession control (steady attack and defense)"
}

# --------------------------------------------------
# New Tactical Map Metrics
# --------------------------------------------------
# 1) Final Third vs Box entries
final3rd_features = [
    "Final Third Pass",
    "Carries into Final Third"
]

box_features = [
    "Pass into Pen Area",
    "Carries into Pen Area"
]

df[final3rd_features] = scaler.fit_transform(df[final3rd_features])
df[box_features] = scaler.fit_transform(df[box_features])

df["Final3rd_Index"] = df[final3rd_features].mean(axis=1)
df["Box_Index"] = df[box_features].mean(axis=1)

# 2) Progressive vs Direct
prog_features = [
    "Prg Carries",
    "Prg Passes",
    "Progressive Receives"
]

direct_features = [
    "Crosses",
    "Cross into Pen Area",
    "Long Pass Att"
]

df[prog_features] = scaler.fit_transform(df[prog_features])
df[direct_features] = scaler.fit_transform(df[direct_features])

df["Progressive_Index"] = df[prog_features].mean(axis=1)
df["Direct_Index"] = df[direct_features].mean(axis=1)

# 3) Pressing vs Compactness
press_high_features = [
    "Tackles Att 3rd",
    "Dribble Stops",
    "Interceptions"
]

compact_def_features = [
    "Tackles Def 3rd",
    "Clearances",
    "Recoveries"
]

df[press_high_features] = scaler.fit_transform(df[press_high_features])
df[compact_def_features] = scaler.fit_transform(df[compact_def_features])

df["HighPress_Index"] = df[press_high_features].mean(axis=1)
df["Compact_Def_Index"] = df[compact_def_features].mean(axis=1)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Attack vs Defense",
    "Build-up vs Attack",
    "Tactical Maps",
    "Cluster & Team Identity"
])

# --------------------------------------------------
# Tab 1: Attack vs Defense map
# --------------------------------------------------
with tab1:
    st.subheader("Attack vs Defense Map")

    st.markdown("""
    **Quadrants:**
    - **Top-right:** Strong attack & strong defense → Dominant teams
    - **Top-left:** Weak attack & strong defense → Low block / defensive teams
    - **Bottom-right:** Strong attack & weak defense → High risk / chaotic teams
    - **Bottom-left:** Weak attack & weak defense → Struggling teams
    """)

    axis_limit = 4
    fig = px.scatter(
        df, x="Attack_Index", y="Defense_Index",
        color="Cluster", hover_name="team",
        title="Attack vs Defense (Tactical Map)"
    )

    fig.add_hline(y=0, line_width=2, line_color="white")
    fig.add_vline(x=0, line_width=2, line_color="white")

    fig.update_xaxes(range=[-axis_limit, axis_limit], fixedrange=True)
    fig.update_yaxes(range=[-axis_limit, axis_limit], fixedrange=True)

    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Tab 2: Build-up vs Attack
# --------------------------------------------------
with tab2:
    st.subheader("Build-up Style vs Attack Strength")

    st.markdown("""
    **Quadrants:**
    - **Top-right:** Strong build-up + strong attack → Possession-based dominant teams
    - **Top-left:** Weak build-up + strong attack → Direct / transition attackers
    - **Bottom-right:** Strong build-up + weak attack → Control-based but low threat
    - **Bottom-left:** Weak build-up + weak attack → Low possession & low threat teams
    """)

    axis_limit2 = 4
    fig2 = px.scatter(
        df, x="Build_Index", y="Attack_Index",
        color="Cluster", hover_name="team",
        title="Build-up Style vs Attack"
    )

    fig2.add_hline(y=0, line_width=2, line_color="white")
    fig2.add_vline(x=0, line_width=2, line_color="white")

    fig2.update_xaxes(range=[-axis_limit2, axis_limit2], fixedrange=True)
    fig2.update_yaxes(range=[-axis_limit2, axis_limit2], fixedrange=True)

    fig2.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=650
    )

    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# Tab 3: Tactical Maps
# --------------------------------------------------
with tab3:
    st.subheader("Tactical Maps")

    def create_map(x_col, y_col, title, x_label, y_label):
        axis_limit = 3.5

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            hover_name="team",
            title=title,
            labels={x_col: x_label, y_col: y_label},
            color="Cluster"
        )

        fig.add_hline(y=0, line_width=2, line_color="white")
        fig.add_vline(x=0, line_width=2, line_color="white")

        fig.update_xaxes(range=[-axis_limit, axis_limit], fixedrange=True)
        fig.update_yaxes(range=[-axis_limit, axis_limit], fixedrange=True)

        fig.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=650
        )
        return fig

    # Map 1: Final Third vs Box Entries
    st.markdown("### 1️⃣ Final Third Activity vs Box Entries")
    st.markdown("""
        **Quadrants:**
        - **Top-right:** High final third activity + high box entries → strong threat and penetration
        - **Top-left:** High final third activity but low box entries → possession-heavy but not penetrating
        - **Bottom-right:** Low final third activity but high box entries → direct, quick attacks
        - **Bottom-left:** Low both → low threat teams
    """)
    st.plotly_chart(
        create_map(
            "Final3rd_Index",
            "Box_Index",
            "Final Third Activity vs Box Entries",
            "Final Third Activity (z-score)",
            "Box Entries (z-score)"
        ),
        use_container_width=True
    )

    # Map 2: Progressive vs Direct
    st.markdown("### 2️⃣ Progressive vs Direct Play")
    st.markdown("""
        **Quadrants:**
        - **Top-right:** High progressive + high direct → elite attackers (both movement and directness)
        - **Top-left:** High progressive but low direct → possession / build-up heavy teams
        - **Bottom-right:** Low progressive but high direct → long ball / wing cross teams
        - **Bottom-left:** Low both → weak transition and attack
    """)
    st.plotly_chart(
        create_map(
            "Progressive_Index",
            "Direct_Index",
            "Progressive vs Direct Play",
            "Progressive Play (z-score)",
            "Direct Play (z-score)"
        ),
        use_container_width=True
    )

    # Map 3: High Press vs Compact Defense
    st.markdown("### 3️⃣ High Press vs Compact Defense")
    st.markdown("""
        **Quadrants:**
        - **Top-right:** High press + strong compact defense → dominant, balanced teams
        - **Top-left:** High press but weak compact defense → high risk, exposed teams
        - **Bottom-right:** Low press but strong compact defense → low block / counter teams
        - **Bottom-left:** Low press + weak defense → vulnerable teams
    """)
    st.plotly_chart(
        create_map(
            "HighPress_Index",
            "Compact_Def_Index",
            "High Press vs Compact Defense",
            "High Press (z-score)",
            "Compact Defense (z-score)"
        ),
        use_container_width=True
    )

# --------------------------------------------------
# Tab 4: Cluster radar + top teams + identity
# --------------------------------------------------
with tab4:
    st.subheader("Cluster Radar + Team Identity")

    # Cluster radar chart (normalized)
    st.markdown("### Cluster Style Radar (Normalized)")

    radar_metrics = [
        "xG", "Shots", "Pass Completion%", "Prg Passes",
        "Touches Att 3rd", "Tackles Att 3rd",
        "Interceptions", "xG Allowed", "Shots on Target Allowed"
    ]

    radar_df = df[radar_metrics].copy()
    for col in radar_metrics:
        min_val = radar_df[col].min()
        max_val = radar_df[col].max()
        radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)

    cluster_centers = radar_df.groupby(df["Cluster"]).mean()

    fig_radar = go.Figure()
    for cluster_id in cluster_centers.index:
        values = cluster_centers.loc[cluster_id].tolist()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics,
            fill="toself",
            name=f"Cluster {cluster_id}"
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Cluster Radar (Normalized Style Signature)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Top 3 teams per cluster
    st.markdown("### Top 3 Teams per Cluster (by Attack Index)")
    top3 = df.sort_values("Attack_Index", ascending=False).groupby("Cluster").head(3)
    st.dataframe(
        top3[["Cluster", "team", "Attack_Index", "Defense_Index", "Build_Index"]]
    )

    # Cluster explanations
    st.markdown("### Cluster Playstyle Explanations")
    for cluster_id, desc in cluster_explanations.items():
        st.write(f"**Cluster {cluster_id}:** {desc}")

    # Tactical Identity Signature
    st.markdown("### Tactical Identity Signature")
    team_select = st.selectbox("Select a team", df["team"].unique())
    team = df[df["team"] == team_select].iloc[0]

    st.write("**Team:**", team_select)
    st.write("**Cluster:**", team["Cluster"])

    st.markdown("#### Key Style Stats")
    st.write({
        "Attack Index": round(team["Attack_Index"], 2),
        "Build-up Index": round(team["Build_Index"], 2),
        "Pressing Index": round(team["Pressing_Index"], 2),
        "Defensive Solidity": round(team["Defensive_Solidity"], 2),
        "Defense Index": round(team["Defense_Index"], 2)
    })

    st.markdown("#### Tactical Summary")
    st.write(
        "• Strong Attack" if team["Attack_Index"] > 1 else "• Average/Weak Attack"
    )
    st.write(
        "• High Build-up" if team["Build_Index"] > 1 else "• Low Build-up"
    )
    st.write(
        "• High Press" if team["Pressing_Index"] > 1 else "• Low Press"
    )
    st.write(
        "• Strong Defense" if team["Defense_Index"] > 1 else "• Weak Defense"
    )
