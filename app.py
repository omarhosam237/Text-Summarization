import streamlit as st
import pandas as pd
from utils import load_cleaned_data, get_word_count

# Page Configuration
st.set_page_config(
    page_title="BART News Article Summarization",
    page_icon="📰",
    layout="wide"
)

# Main Header
st.title("📰 BART News Article Summarization")
st.caption("Graduation Project | Natural Language Processing & Sequence-to-Sequence Modeling")

st.markdown("""
Welcome to the interactive web application for **Abstractive News Text Summarization** using a fine-tuned 
**BART (Bidirectional and Auto-Regressive Transformers)** model. 

This project takes long news articles from the benchmark **CNN/DailyMail** dataset and generates concise, fluent, 
and contextually accurate abstractive summaries.
""")

st.divider()

# Section 1: Overview & Methodology
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📌 Project Overview & Pipeline")
    st.markdown("""
    - **Dataset**: CNN/DailyMail 3.0.0 (`abisee/cnn_dailymail`), initialized with 10,000 training samples and filtered to quality article lengths (100–1,000 words) and summary lengths (10–100 words).
    - **Base Pretrained Model**: `facebook/bart-large-cnn`
    - **Fine-Tuned Model**: [`omarhosam-237/bart-news-summarizer`](https://huggingface.co/omarhosam-237/bart-news-summarizer)
    - **Task Type**: **Abstractive Summarization** (paraphrases and generates new sentences rather than extracting verbatim lines).
    - **Training Strategy**: Fine-tuned using Hugging Face `Seq2SeqTrainer` with Cross-Entropy Loss & ROUGE evaluation metrics.
    """)

with col2:
    st.subheader("💡 Key Concepts Explained")
    st.markdown("""
    - **BART Architecture**: Combines a bidirectional encoder (like BERT) with an autoregressive decoder (like GPT), making it suited for Seq2Seq generation.
    - **Tokenization**: Uses Byte-Pair Encoding (BPE) tokenization with max input sequence length of 1,024 tokens.
    - **Beam Search Decoding**: Generates summaries using candidate beam paths (beam size = 4) to ensure high generation quality and coherence.
    """)

st.divider()

# Load cleaned dataset for dynamic stats
df_clean = load_cleaned_data("cleaned_data.csv")

# Section 2: Dataset Statistics
st.subheader("📊 Dataset Statistics (CNN/DailyMail Processed Subset)")

if not df_clean.empty:
    total_rows = len(df_clean)
    train_rows = 6365 if total_rows == 7957 else int(total_rows * 0.8)
    test_rows = total_rows - train_rows
    avg_art_len = int(df_clean['article_length'].mean())
    avg_sum_len = int(df_clean['highlights_length'].mean())
    
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)
    col_d1.metric(label="Total Clean Samples", value=f"{total_rows:,}")
    col_d2.metric(label="Training Split (80%)", value=f"{train_rows:,}")
    col_d3.metric(label="Test Split (20%)", value=f"{test_rows:,}")
    col_d4.metric(label="Avg Article Length", value=f"{avg_art_len} words")
    col_d5.metric(label="Avg Summary Length", value=f"{avg_sum_len} words")
else:
    st.warning("Cleaned dataset not available. Displaying default processed sample stats.")
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)
    col_d1.metric(label="Total Clean Samples", value="7,957")
    col_d2.metric(label="Training Split (80%)", value="6,365")
    col_d3.metric(label="Test Split (20%)", value="1,592")
    col_d4.metric(label="Avg Article Length", value="543 words")
    col_d5.metric(label="Avg Summary Length", value="44 words")

st.divider()

# Section 3: Final Training & Evaluation Metrics
st.subheader("🏆 Final Fine-Tuning Evaluation Results")
st.markdown("Below are the exact metrics from the final training run of **`omarhosam-237/bart-news-summarizer`**:")

m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
m_col1.metric(label="Training Loss", value="0.5471")
m_col2.metric(label="Validation Loss", value="0.5361")
m_col3.metric(label="ROUGE-1", value="41.74")
m_col4.metric(label="ROUGE-2", value="19.82")
m_col5.metric(label="ROUGE-L", value="29.58")
m_col6.metric(label="ROUGE-Lsum", value="39.29")

st.caption("ROUGE scores represent n-gram overlap and sentence-level similarity between generated summaries and ground-truth human reference summaries.")

st.divider()

# Section 4: Application Navigation
st.subheader("🚀 Explore the Application")
nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.info("""
    ### 📊 Page 1: Dataset Analysis
    Explore interactive distributions, box plots, heatmap correlations, and sample rows from the cleaned CNN/DailyMail dataset.
    """)

with nav_col2:
    st.success("""
    ### ✍️ Page 2: Text Summarization
    Test the fine-tuned BART model live! Paste custom news text or choose from sample articles to generate instant summaries.
    """)
