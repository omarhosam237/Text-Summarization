import streamlit as st
from utils import (
    load_model_and_tokenizer,
    load_cleaned_data,
    generate_summary,
    get_word_count,
    compute_metrics
)

st.set_page_config(
    page_title="Text Summarization | BART Model",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Abstractive Text Summarization")
st.markdown("Generate concise news summaries using the fine-tuned **`omarhosam-237/bart-news-summarizer`** model.")

# Load fine-tuned model and tokenizer
with st.spinner("Loading BART fine-tuned model from Hugging Face Hub (cached)..."):
    try:
        tokenizer, model, device = load_model_and_tokenizer()
        st.sidebar.success(f"Model loaded successfully on `{device.type.upper()}`!")
    except Exception as e:
        st.error("Failed to load model. Please check your internet connection or Hugging Face access.")
        st.stop()

# Generation parameters in sidebar for transparency
st.sidebar.header("⚙️ Generation Parameters")
st.sidebar.markdown("""
These parameters guide BART's beam search decoder:
- **Max Summary Length**: Maximum tokens (128)
- **Min Summary Length**: Minimum tokens (30)
- **Beam Search Width**: Candidate paths evaluated (num_beams=4)
- **Truncation Limit**: Articles truncated beyond 1,024 tokens
""")

num_beams = st.sidebar.slider("Beam Search Width (num_beams):", min_value=1, max_value=8, value=4)
max_len = st.sidebar.slider("Max Summary Length:", min_value=30, max_value=256, value=128)
min_len = st.sidebar.slider("Min Summary Length:", min_value=10, max_value=100, value=30)

# Input Tabs: Custom Text vs Sample Articles
tab1, tab2 = st.tabs(["📝 Input Custom News Article", "📰 Try Sample Articles from Dataset"])

# =========================================================
# TAB 1: CUSTOM TEXT SUMMARIZATION
# =========================================================
with tab1:
    st.subheader("Paste your News Article")
    
    default_text = (
        "LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported "
        "20 million dollar fortune as he turns 18 on Monday, but he insists the money will not cast "
        "a spell on him. Radcliffe, who plays the wizard boy in the film adaptations of J.K. Rowling "
        "books, said he had no immediate plans to spend the cash on high-end luxury items. "
        "Friends and colleagues say the actor remains grounded despite his enormous international success."
    )
    
    user_input = st.text_area(
        label="Enter news text below:",
        value=default_text,
        height=220,
        placeholder="Paste news article here..."
    )
    
    # Real-time metrics
    char_count = len(user_input)
    word_count = get_word_count(user_input)
    
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.info(f"**Word Count:** {word_count} words")
    col_info2.info(f"**Character Count:** {char_count} chars")
    
    # Input validation warnings
    if word_count > 0 and word_count < 50:
        st.warning("⚠️ **Short Article Warning:** Input text is very short (< 50 words). Abstractive BART model performs best on standard news articles.")
    elif word_count > 800:
        st.info("ℹ️ **Truncation Notice:** Input exceeds 1,024 token limit. Text will be safely truncated during tokenization.")
        
    generate_btn = st.button("✨ Generate Summary", type="primary", key="btn_custom")
    
    if generate_btn:
        if not user_input.strip():
            st.error("Please enter a valid news article before generating a summary.")
        else:
            with st.spinner("Generating summary using BART model..."):
                summary_text = generate_summary(
                    article_text=user_input,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=max_len,
                    min_length=min_len,
                    num_beams=num_beams
                )
                
            st.success("Summary generated successfully!")
            
            st.markdown("### 📋 Generated Summary")
            st.success(f"**{summary_text}**")
            
            # Compute reduction statistics
            metrics = compute_metrics(user_input, summary_text)
            
            st.markdown("### 📊 Compression & Length Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Original Word Count", metrics["orig_words"])
            m_col2.metric("Summary Word Count", metrics["summ_words"])
            m_col3.metric("Compression Ratio", f"{metrics['compression_ratio']}x")
            m_col4.metric("Length Reduction", f"{metrics['reduction_percentage']}%")

# =========================================================
# TAB 2: SAMPLE ARTICLE MODE
# =========================================================
with tab2:
    st.subheader("Compare Reference vs. Model Summary")
    st.markdown("Select a real article from the cleaned **CNN/DailyMail** test dataset to evaluate generated quality.")
    
    df_clean = load_cleaned_data("cleaned_data.csv")
    
    if df_clean.empty:
        st.error("Dataset 'cleaned_data.csv' unavailable for sample selection.")
    else:
        # Pick 10 sample indices
        sample_indices = list(range(min(10, len(df_clean))))
        sample_labels = [f"Sample #{i+1} ({df_clean.iloc[i]['article_length']} words)" for i in sample_indices]
        
        selected_sample_idx = st.selectbox(
            "Choose a sample article:",
            options=sample_indices,
            format_func=lambda i: sample_labels[i]
        )
        
        sample_row = df_clean.iloc[selected_sample_idx]
        sample_article = sample_row["article"]
        reference_summary = sample_row["highlights"]
        
        st.divider()
        
        # Display Original Article
        st.markdown(f"### 📰 Original Article (Sample #{selected_sample_idx+1})")
        st.caption(f"Length: {sample_row['article_length']} words")
        st.text_area("Full Article Text:", value=sample_article, height=180, disabled=True)
        
        btn_sample_gen = st.button("✨ Generate Summary for Selected Sample", type="primary", key="btn_sample")
        
        if btn_sample_gen or f"generated_sample_{selected_sample_idx}" in st.session_state:
            if btn_sample_gen:
                with st.spinner("Generating summary for sample article..."):
                    gen_summary = generate_summary(
                        article_text=sample_article,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        max_length=max_len,
                        min_length=min_len,
                        num_beams=num_beams
                    )
                    st.session_state[f"generated_sample_{selected_sample_idx}"] = gen_summary
            else:
                gen_summary = st.session_state[f"generated_sample_{selected_sample_idx}"]
                
            st.divider()
            st.markdown("### 🔍 Summary Comparison")
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.info("🎯 **Reference Summary (Human Ground Truth)**")
                st.write(reference_summary)
                st.caption(f"Word count: {get_word_count(reference_summary)} words")
                
            with comp_col2:
                st.success("🤖 **Generated Summary (BART Model Output)**")
                st.write(gen_summary)
                st.caption(f"Word count: {get_word_count(gen_summary)} words")
                
            # Metrics for sample comparison
            metrics = compute_metrics(sample_article, gen_summary)
            st.markdown("#### 📊 Sample Performance Stats")
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.metric("Article Word Count", metrics["orig_words"])
            s_col2.metric("Generated Word Count", metrics["summ_words"])
            s_col3.metric("Compression Ratio", f"{metrics['compression_ratio']}x")
            s_col4.metric("Length Reduction", f"{metrics['reduction_percentage']}%")
