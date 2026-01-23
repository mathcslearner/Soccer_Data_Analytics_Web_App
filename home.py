# Home page for the Streamlit Soccer Data Analytics App

import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------
# Page config
# --------------------
st.set_page_config(
    page_title="European Soccer Analytics",
    layout="wide"
)

# --------------------
# Load data
# --------------------
@st.cache_data
def load_data():
    return pd.read_csv("team_stats_2024-2025.csv")

df = load_data()

# --------------------
# Title
# --------------------
st.title("European Top 5 Leagues - Team Performance Analytics")
st.markdown(
    "Interactive analysis of **attacking**, **defensive**, **possession** and other advanced metrics "
    "across Europe's top five leagues."
)

st.divider()

# --------------------
# Sidebar filters
# --------------------
st.sidebar.header("Filters")

league = st.sidebar.multiselect(
    "Select League(s)",
    options=sorted(df["league"].unique()),
    default=sorted(df["league"].unique())
)

season = st.sidebar.multiselect(
    "Select Season(s)",
    options=sorted(df["season"].unique()),
    default=[df["season"].max()]
)

filtered_df = df[
    (df["league"].isin(league)) &
    (df["season"].isin(season))
]

# --------------------
# KPI metrics
# --------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Avg Goals", round(filtered_df["Goals"].mean(), 2))

with col2:
    st.metric("Avg xG", round(filtered_df["xG"].mean(), 2))

with col3:
    st.metric("Avg Possession %", round(filtered_df["Poss"].mean(), 1))

with col4:
    st.metric("Avg Goal Difference", round(filtered_df["Goal Difference"].mean(), 2))

with col5:
    st.metric("Avg Clean Sheet %", round(filtered_df["Clean Sheet%"].mean(), 1))

st.divider()

# --------------------
# League comparison charts
# --------------------
st.subheader("League Comparison")

league_summary = (
    filtered_df
    .groupby("league")
    .agg({
        "Goals": "mean",
        "xG": "mean",
        "Poss": "mean"
    })
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:
    fig_goals = px.bar(
        league_summary,
        x="league",
        y="Goals",
        title="Average Goals per Team",
        labels={"Goals": "Goals"}
    )
    st.plotly_chart(fig_goals, use_container_width=True)
    st.caption("Insight: The EPL is a more attacking-minded league, while Serie A is a more defensive-minded league")

with col2:
    fig_xg = px.bar(
        league_summary,
        x="league",
        y="xG",
        title="Average Expected Goals (xG)",
        labels={"xG": "xG"}
    )
    st.plotly_chart(fig_xg, use_container_width=True)
    st.caption("Insight: In all league except Ligue 1, teams overperform their xG on average! The league where teams overperform the most is La Liga.")

st.divider()

# --------------------
# Goals vs xG scatter
# --------------------
st.subheader("Goals vs Expected Goals")

fig_scatter = px.scatter(
    filtered_df,
    x="xG",
    y="Goals",
    color="league",
    hover_name="team",
    title="Goals vs xG (Over / Underperformance)",
    labels={"xG": "Expected Goals", "Goals": "Goals Scored"}
)

# Get axis range
max_val = max(filtered_df["xG"].max(), filtered_df["Goals"].max())

# Add y = x line
fig_scatter.add_shape(
    type="line",
    x0=0, y0=0,
    x1=max_val, y1=max_val,
    line=dict(dash="dash", color = "rgba(220, 220, 220, 0.9)")
)

fig_scatter.add_annotation(
    x=max_val * 0.95,
    y=max_val * 0.95,
    text="Goals = xG",
    showarrow=False,
    font=dict(size=12)
)

st.plotly_chart(fig_scatter, use_container_width=True)
st.caption("Insight: As noted above, teams tend to overperform more than underperform on their xG")

st.divider()

# ----------------
# Top xG overperformers and underperformers
# ----------------

xg_df = filtered_df.copy()

# Avoid division by zero
xg_df = xg_df[xg_df["xG"] > 0]

xg_df["xG Overperformance %"] = (
    (xg_df["Goals"] - xg_df["xG"]) / xg_df["xG"]
) * 100

st.subheader("xG Over / Underperformance (%)")

col1, col2 = st.columns(2)

# -------- Top 5 Overperformers --------
with col1:
    st.markdown("### Top 5 xG Overperformers")

    top_over = (
        xg_df
        .sort_values("xG Overperformance %", ascending=False)
        .loc[:, ["team", "league", "Goals", "xG", "xG Overperformance %"]]
        .head(5)
        .reset_index(drop=True)
    )

    top_over.index += 1
    top_over["xG Overperformance %"] = top_over["xG Overperformance %"].round(1)

    styled_top_over = (
    top_over
    .style
    .background_gradient(
        subset=["xG Overperformance %"],
        cmap="RdYlGn"
    )
    .format({
        "xG": "{:.1f}",
        "xG Overperformance %": "{:.1f}"
    })
)

    st.dataframe(
        styled_top_over,
        use_container_width=True
    )

st.caption("Insight: xG overperformance and underperformance tend to regress to the mean over time, so this metric identifies teams whose performance might be unsustainable")

# -------- Top 5 Underperformers --------
with col2:
    st.markdown("### Top 5 xG Underperformers")

    top_under = (
        xg_df
        .sort_values("xG Overperformance %", ascending=True)
        .loc[:, ["team", "league", "Goals", "xG", "xG Overperformance %"]]
        .head(5)
        .reset_index(drop=True)
    )

    top_under.index += 1
    top_under["xG Overperformance %"] = top_under["xG Overperformance %"].round(1)

    styled_top_under = (
    top_under
    .style
    .background_gradient(
        subset=["xG Overperformance %"],
        cmap="RdYlGn"
    )
    .format({
        "xG": "{:.1f}",
        "xG Overperformance %": "{:.1f}"
    })
)

    st.dataframe(
        styled_top_under,
        use_container_width=True
    )



# --------------------
# Top 5 teams by category
# ---------------------

st.subheader("Top 5 Teams by Key Metrics")

def top5_table(df, metric, ascending=False, decimals=2):
    top5 = (
        df.sort_values(metric, ascending=ascending)
        .loc[:, ["team", "league", metric]]
        .head(5)
        .reset_index(drop=True)
    )
    top5.index += 1
    top5[metric] = top5[metric].round(decimals)
    return top5

# -------- Row 1 (3 columns) --------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Pts / Match")
    st.dataframe(
        top5_table(filtered_df, "Pts pMatch"),
        use_container_width=True,
        hide_index=False
    )

with col2:
    st.markdown("### Goals Scored")
    st.dataframe(
        top5_table(filtered_df, "Goals", decimals=0),
        use_container_width=True,
        hide_index=False
    )

with col3:
    st.markdown("### Expected Goals (xG)")
    st.dataframe(
        top5_table(filtered_df, "xG"),
        use_container_width=True,
        hide_index=False
    )

# -------- Row 2 (2 columns, centered) --------
spacer, col4, col5, spacer2 = st.columns([0.5, 1.2, 1, 0.5])

with col4:
    st.markdown("### Goal Difference")
    st.dataframe(
        top5_table(filtered_df, "Goal Difference", decimals=0),
        use_container_width=True,
        hide_index=False
    )

with col5:
    st.markdown("### Possession %")
    st.dataframe(
        top5_table(filtered_df, "Poss", decimals=1),
        use_container_width=True,
        hide_index=False
    )


# --------------------
# Top teams table
# --------------------
st.subheader("Top Teams Snapshot")

metric_options = {
    "Points per Match": "Pts pMatch",
    "Goals Scored": "Goals",
    "Expected Goals (xG)": "xG",
    "Goal Difference": "Goal Difference",
    "Possession %": "Poss"
}

selected_metric_label = st.selectbox(
    "Rank teams by",
    options=list(metric_options.keys()),
    index=0
)

st.markdown(f"**Top 10 teams by {selected_metric_label}**")

selected_metric = metric_options[selected_metric_label]

snapshot_leagues = st.multiselect(
    "Limit snapshot to league(s)",
    options=filtered_df["league"].unique(),
    default=filtered_df["league"].unique()
)

snapshot_df = filtered_df[
    filtered_df["league"].isin(snapshot_leagues)
]

# Sort direction (possession & goals are higher-is-better)
top10 = (
    filtered_df
    .sort_values(selected_metric, ascending=False)
    .loc[:, [
        "team",
        "league",
        "Pts pMatch",
        "Goals",
        "xG",
        "Goal Difference",
        "Poss"
    ]]
    .head(10)
    .reset_index(drop=True)
)

top10.index += 1

st.dataframe(
    top10.style.background_gradient(
        subset=[selected_metric],
        cmap="viridis"
    ),
    use_container_width=True
)


# --------------------
# Footer hint
# --------------------
st.info(
    "Use the sidebar to filter leagues and seasons, or explore deeper analysis "
    "from the navigation menu."
)
