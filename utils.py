import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "omarhosam-237/bart-news-summarizer"

@st.cache_resource
def load_model_and_tokenizer():
    """
    Cache and load the fine-tuned BART tokenizer and model from Hugging Face Hub.
    Uses GPU acceleration if CUDA is available, otherwise CPU.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Failed to load model '{MODEL_ID}' from Hugging Face Hub: {e}")
        raise e

@st.cache_data
def load_cleaned_data(file_path: str = "cleaned_data.csv") -> pd.DataFrame:
    """
    Cache and load the preprocessed CNN/DailyMail dataset from local CSV.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Cleaned dataset file '{file_path}' not found. Please ensure cleaned_data.csv exists.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

def generate_summary(article_text: str, tokenizer, model, device, max_length: int = 128, min_length: int = 30, num_beams: int = 4) -> str:
    """
    Generates an abstractive summary using the fine-tuned BART model.
    
    Parameters:
    - article_text: Input news article
    - tokenizer: BART AutoTokenizer
    - model: Fine-tuned AutoModelForSeq2SeqLM
    - device: torch device ('cuda' or 'cpu')
    - max_length: maximum token length of generated summary (default: 128)
    - min_length: minimum token length of generated summary (default: 30)
    - num_beams: beam search count for decoding (default: 4)
    """
    if not article_text or not article_text.strip():
        return ""
        
    inputs = tokenizer(
        article_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()

def get_word_count(text: str) -> int:
    """Helper function to calculate word count."""
    if not text:
        return 0
    return len(str(text).split())

def compute_metrics(original_text: str, summary_text: str):
    """
    Calculates word counts, compression ratio, and percentage reduction safely.
    """
    orig_words = get_word_count(original_text)
    summ_words = get_word_count(summary_text)
    
    if orig_words > 0:
        compression_ratio = round(orig_words / max(summ_words, 1), 2)
        reduction_percentage = round(((orig_words - summ_words) / orig_words) * 100, 1)
    else:
        compression_ratio = 0.0
        reduction_percentage = 0.0
        
    return {
        "orig_words": orig_words,
        "summ_words": summ_words,
        "compression_ratio": compression_ratio,
        "reduction_percentage": max(0.0, reduction_percentage)
    }
