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
    Analyze team style through Attack vs Defense, Build-up vs Attack, cluster radar, and team identity.
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
# Tabs
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Attack vs Defense", "Build-up vs Attack", "Cluster & Team Identity"])

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
# Tab 3: Cluster radar + top teams + identity
# --------------------------------------------------
with tab3:
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
