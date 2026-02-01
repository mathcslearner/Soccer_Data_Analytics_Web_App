# Playground page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import chi2 

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
    st.subheader("Interactive Plot Builder with 2D Outlier Highlighting")

    numeric_df = df.select_dtypes(include="number")

    col1, col2, col3 = st.columns(3)

    with col1:
        x_axis = st.selectbox("X-axis", numeric_df.columns, key="x_axis_maha")

    with col2:
        y_axis = st.selectbox("Y-axis", numeric_df.columns, key="y_axis_maha")

    with col3:
        plot_type = st.selectbox("Plot Type", ["Scatter", "Line", "Bar"], key="plot_type_maha")

    if st.button("Generate Plot (2D Outliers)"):
        if plot_type == "Scatter":

            # -------- Mahalanobis Distance Outlier Detection --------
            data = df[[x_axis, y_axis]].dropna()
            cov_matrix = np.cov(data.values, rowvar=False)
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
            mean_vals = data.mean().values

            # compute Mahalanobis distance
            diff = data.values - mean_vals
            md = np.sqrt(np.sum(diff @ inv_cov_matrix * diff, axis=1))

            # Threshold for outliers (chi-square, 2 degrees of freedom)
            threshold = np.sqrt(chi2.ppf(0.975, df=2))  # 97.5% confidence
            outlier_mask = md > threshold

            df_plot = df.copy()
            df_plot["Outlier"] = outlier_mask

            # -------- Scatter Plot --------
            fig = px.scatter(
                df_plot,
                x=x_axis,
                y=y_axis,
                color="Outlier",               # outliers in red
                hover_name="team",
                hover_data=["league", "season", x_axis, y_axis],
                title=f"{x_axis} vs {y_axis} (2D outliers highlighted)",
                color_discrete_map={True: "red", False: "blue"},
                opacity=0.7,
                size=[15 if o else 8 for o in df_plot["Outlier"]]
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

    # Select target variable
    target = st.selectbox(
        "Select a target variable (what you care about predicting/explaining)",
        numeric_df.columns
    )

    # Compute correlations with target
    corr = numeric_df.corr()[target].dropna().sort_values(ascending=False)

    # Strongest positive correlations (exclude self-correlation)
    top_pos = corr[corr.index != target].head(10)

    # Strongest negative correlations
    top_neg = corr.tail(10)

    # Display in tables
    st.markdown("### Strongest Positive Relationships")
    st.dataframe(top_pos.to_frame("Correlation"))

    st.markdown("### Strongest Negative Relationships")
    st.dataframe(top_neg.to_frame("Correlation"))

    # Combine for plotting
    plot_df = pd.concat([top_pos, top_neg]).reset_index()
    plot_df.columns = ["Feature", "Correlation"]

    # Horizontal bar plot
    import plotly.express as px

    fig = px.bar(
        plot_df,
        x="Correlation",
        y="Feature",
        orientation="h",
        text="Correlation",
        title=f"Top Relationships with {target}"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Auto Insights Text
    st.markdown("### Auto Insights")
    for feature, value in top_pos.head(3).items():
        st.write(f"• **{feature}** strongly increases with **{target}** (corr = {value:.2f})")
    for feature, value in top_neg.head(3).items():
        st.write(f"• **{feature}** strongly decreases with **{target}** (corr = {value:.2f})")


