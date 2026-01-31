# Playground page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Football Data Playground", layout="wide")

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("team_stats_2024-2025.csv")

df = load_data()

st.title("Football Analytics Playground")
st.write("Interactive playground for exploring football performance data")

st.success("Dataset loaded successfully!")

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filters")

if "league" in df.columns:
    leagues = st.sidebar.multiselect(
        "League",
        sorted(df["league"].dropna().unique())
    )
    if leagues:
        df = df[df["league"].isin(leagues)]

if "season" in df.columns:
    seasons = st.sidebar.multiselect(
        "Season",
        sorted(df["season"].dropna().unique())
    )
    if seasons:
        df = df[df["season"].isin(seasons)]

if "team" in df.columns:
    teams = st.sidebar.multiselect(
        "Team",
        sorted(df["team"].dropna().unique())
    )
    if teams:
        df = df[df["team"].isin(teams)]

# =========================
# Tabs Layout
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Explorer",
    "Stats Lab",
    "Visual Playground",
    "Correlations"
])

# =========================
# Data Explorer
# =========================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Column Viewer")
    selected_cols = st.multiselect(
        "Select columns to inspect",
        df.columns.tolist()
    )
    if selected_cols:
        st.dataframe(df[selected_cols], use_container_width=True)

# =========================
# Stats Lab
# =========================
with tab2:
    st.subheader("Summary Statistics")
    numeric_df = df.select_dtypes(include="number")
    st.dataframe(numeric_df.describe().T, use_container_width=True)

# =========================
# Visual Playground
# =========================
with tab3:
    st.subheader("Interactive Plot Builder")

    numeric_df = df.select_dtypes(include="number")

    col1, col2, col3 = st.columns(3)

    with col1:
        x_axis = st.selectbox("X-axis", numeric_df.columns)

    with col2:
        y_axis = st.selectbox("Y-axis", numeric_df.columns)

    with col3:
        plot_type = st.selectbox("Plot Type", ["Scatter", "Line", "Bar"])

    if st.button("Generate Plot"):

        if plot_type == "Scatter":
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                hover_name="team",   # 👈 hover shows team
                hover_data=["league", "season"],
                title=f"{x_axis} vs {y_axis}",
            )

        elif plot_type == "Line":
            fig = px.line(
                df,
                x=x_axis,
                y=y_axis,
                title=f"{x_axis} vs {y_axis}",
            )

        elif plot_type == "Bar":
            fig = px.bar(
                df,
                x=x_axis,
                y=y_axis,
                title=f"{x_axis} vs {y_axis}",
            )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# Insightful Correlation Lab
# =========================
with tab4:
    st.subheader("Relationship Explorer (Insight Engine)")

    numeric_df = df.select_dtypes(include="number")

    target = st.selectbox(
        "Select a target variable (what you care about predicting/explaining)",
        numeric_df.columns
    )

    corr = numeric_df.corr()[target].dropna().sort_values(ascending=False)

    st.markdown("### Strongest Positive Relationships")
    top_pos = corr[1:11]  # skip self-correlation
    st.dataframe(top_pos.to_frame("Correlation"))

    st.markdown("### Strongest Negative Relationships")
    top_neg = corr[-10:]
    st.dataframe(top_neg.to_frame("Correlation"))

    # -------- Plot --------
    st.markdown("### Visual Relationships")

    top_features = pd.concat([top_pos, top_neg]).index.tolist()

    fig = px.bar(
        x=corr[top_features].values,
        y=top_features,
        orientation="h",
        labels={"x": "Correlation Strength", "y": "Feature"},
        title=f"Top Relationships with {target}"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------- Auto Insight Text --------
    st.markdown("### Auto Insights")

    insight_text = []
    for feature, value in top_pos.head(3).items():
        insight_text.append(f"• **{feature}** strongly increases with **{target}** (corr = {value:.2f})")

    for feature, value in top_neg.head(3).items():
        insight_text.append(f"• **{feature}** strongly decreases with **{target}** (corr = {value:.2f})")

    for line in insight_text:
        st.write(line)

