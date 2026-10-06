import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_cleaned_data

st.set_page_config(
    page_title="Dataset Analysis | BART Summarization",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dataset Analysis")
st.markdown("Exploratory Data Analysis (EDA) of the preprocessed **CNN/DailyMail 3.0.0** dataset subset.")

# Load dataset
df_clean = load_cleaned_data("cleaned_data.csv")

if df_clean.empty:
    st.error("Dataset 'cleaned_data.csv' could not be loaded. Please ensure it exists in the root directory.")
    st.stop()

# Sidebar Filters
st.sidebar.header("🔍 Dataset Filters")

min_art_len = int(df_clean["article_length"].min())
max_art_len = int(df_clean["article_length"].max())
min_hl_len = int(df_clean["highlights_length"].min())
max_hl_len = int(df_clean["highlights_length"].max())

art_len_range = st.sidebar.slider(
    "Article Length (words):",
    min_value=min_art_len,
    max_value=max_art_len,
    value=(min_art_len, max_art_len),
    step=10
)

hl_len_range = st.sidebar.slider(
    "Summary / Highlights Length (words):",
    min_value=min_hl_len,
    max_value=max_hl_len,
    value=(min_hl_len, max_hl_len),
    step=5
)

# Filter Dataframe
df_filtered = df_clean[
    (df_clean["article_length"] >= art_len_range[0]) &
    (df_clean["article_length"] <= art_len_range[1]) &
    (df_clean["highlights_length"] >= hl_len_range[0]) &
    (df_clean["highlights_length"] <= hl_len_range[1])
].copy()

# Filtered Summary Metrics
st.subheader("📈 Subset Summary Statistics")
col1, col2, col3, col4, col5 = st.columns(5)

filtered_count = len(df_filtered)
avg_art = round(df_filtered["article_length"].mean(), 1) if filtered_count > 0 else 0
med_art = round(df_filtered["article_length"].median(), 1) if filtered_count > 0 else 0
avg_hl = round(df_filtered["highlights_length"].mean(), 1) if filtered_count > 0 else 0
med_hl = round(df_filtered["highlights_length"].median(), 1) if filtered_count > 0 else 0

col1.metric("Filtered Samples", f"{filtered_count:,}")
col2.metric("Avg Article Length", f"{avg_art} words")
col3.metric("Median Article Length", f"{med_art} words")
col4.metric("Avg Summary Length", f"{avg_hl} words")
col5.metric("Median Summary Length", f"{med_hl} words")

st.divider()

if filtered_count == 0:
    st.warning("No samples match the selected filter criteria. Please broaden your slider ranges.")
    st.stop()

# Visualizations Section
st.subheader("🖼️ Interactive Visualizations")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # 1. Article Length Distribution Histogram
    fig_art = px.histogram(
        df_filtered,
        x="article_length",
        nbins=30,
        title="Distribution of Article Lengths",
        color_discrete_sequence=["#1f77b4"],
        labels={"article_length": "Article Word Count", "count": "Frequency"}
    )
    fig_art.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_art, use_container_width=True)

with chart_col2:
    # 2. Highlights Length Distribution Histogram
    fig_hl = px.histogram(
        df_filtered,
        x="highlights_length",
        nbins=25,
        title="Distribution of Highlights / Summary Lengths",
        color_discrete_sequence=["#2ca02c"],
        labels={"highlights_length": "Summary Word Count", "count": "Frequency"}
    )
    fig_hl.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_hl, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    # 3. Article vs Summary Length Scatter Plot
    fig_scatter = px.scatter(
        df_filtered,
        x="article_length",
        y="highlights_length",
        opacity=0.6,
        title="Article Length vs. Highlights Length",
        color_discrete_sequence=["#ff7f0e"],
        labels={"article_length": "Article Word Count", "highlights_length": "Summary Word Count"}
    )
    fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

with chart_col4:
    # 4. Density Heatmap
    fig_heatmap = px.density_heatmap(
        df_filtered,
        x="article_length",
        y="highlights_length",
        title="Heatmap of Article Length vs. Summary Length",
        color_continuous_scale="Viridis",
        labels={"article_length": "Article Word Count", "highlights_length": "Summary Word Count"}
    )
    fig_heatmap.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# Dataset Sample Preview Table
st.subheader("📄 Dataset Preview")
st.markdown("Inspect sample rows from the cleaned dataset (showing up to 5 rows).")

preview_rows = st.slider("Number of preview rows:", min_value=1, max_value=10, value=5)
df_display = df_filtered.head(preview_rows)

for idx, row in df_display.iterrows():
    with st.expander(f"📌 Sample #{idx+1} | Article: {row['article_length']} words | Summary: {row['highlights_length']} words"):
        st.markdown("**Article Preview:**")
        st.write(row['article'])
        st.markdown("**Reference Summary (Ground Truth):**")
        st.info(row['highlights'])
