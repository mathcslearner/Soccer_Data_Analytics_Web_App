# Playground page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Explorer",
    "Stats Lab",
    "Visual Playground",
    "Correlations",
    "Feature Importance",
    "PCA / UMAP"
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

# =============
# Feature Importance (ML)
# ==============
with tab5:
    st.subheader("Feature Importance Engine")

    numeric_df = df.select_dtypes(include="number")
    target = st.selectbox("Select prediction target", numeric_df.columns, key="fi_target")

    X = numeric_df.drop(columns=[target]).fillna(0)
    y = numeric_df[target].fillna(0)

    if st.button("Train Model & Compute Importance"):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        fi_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.markdown("### 🔝 Most Important Features")
        st.dataframe(fi_df.head(15))

        fig = px.bar(
            fi_df.head(20),
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top Predictors of {target}",
            text="Importance"
        )
        st.plotly_chart(fig, use_container_width=True)

        r2 = rf.score(X_test, y_test)
        st.success(f"Model performance (R² on test set): {r2:.3f}")

# ===========
# PCA/UMAP tab
# ===========

with tab6:
    st.subheader("PCA / UMAP Tactical Space")

    numeric_df = df.select_dtypes(include="number")

    st.markdown("### Feature Selection")
    selected_features = st.multiselect(
        "Select features for embedding (high-dimensional space)",
        numeric_df.columns.tolist(),
        default=list(numeric_df.columns[:20])  # safe default
    )

    if len(selected_features) < 3:
        st.warning("Select at least 3 features for PCA/UMAP")
    else:
        X = numeric_df[selected_features].fillna(0)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        col1, col2 = st.columns(2)

        with col1:
            method = st.selectbox("Embedding Method", ["PCA", "UMAP"])

        with col2:
            color_by = st.selectbox(
                "Color points by",
                ["None"] + [c for c in ["league", "season"] if c in df.columns]
            )

        if method == "PCA":
            pca = PCA(n_components=2, random_state=42)
            embedding = pca.fit_transform(X_scaled)
            explained = pca.explained_variance_ratio_

            df_embed = df.copy()
            df_embed["Dim1"] = embedding[:, 0]
            df_embed["Dim2"] = embedding[:, 1]

            title = f"PCA Tactical Map (Explained Var: {explained[0]:.2f}, {explained[1]:.2f})"

        else:  # UMAP
            n_neighbors = st.slider("UMAP n_neighbors", 5, 50, 15)
            min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1)

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42
            )
            embedding = reducer.fit_transform(X_scaled)

            df_embed = df.copy()
            df_embed["Dim1"] = embedding[:, 0]
            df_embed["Dim2"] = embedding[:, 1]

            title = "UMAP Tactical Map (Nonlinear Structure)"

        # ---------- Plot ----------
        fig = px.scatter(
            df_embed,
            x="Dim1",
            y="Dim2",
            color=None if color_by == "None" else color_by,
            hover_name="team",
            hover_data=["league", "season"],
            title=title,
            opacity=0.8
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------- Interpretation Panel ----------
        st.markdown("### 🧠 Interpretation Guide")
        if method == "PCA":
            st.write("""
            • Distance = statistical similarity  
            • Clusters = similar team profiles  
            • Direction = tactical/statistical gradients  
            • Axes = linear combinations of features  
            """)
        else:
            st.write("""
            • Distance = structural similarity  
            • Clusters = playstyle archetypes  
            • Outliers = unique tactical identities  
            • Nonlinear structure = complex patterns  
            """)
