# 📰 BART Abstractive News Article Summarization

A professional multi-page Streamlit web application for abstractive news article summarization using a fine-tuned **BART (Bidirectional and Auto-Regressive Transformers)** sequence-to-sequence model.

---

## 📌 1. Project Overview

Abstractive summarization creates fluent, concise summaries by generating novel sentences rather than extracting verbatim snippets. This project fine-tunes **`facebook/bart-large-cnn`** on the **CNN/DailyMail** news dataset to generate abstractive summaries for long news articles.

- **Hugging Face Model Hub ID**: [`omarhosam-237/bart-news-summarizer`](https://huggingface.co/omarhosam-237/bart-news-summarizer)
- **Frameworks**: PyTorch, Hugging Face `transformers`, Streamlit, Plotly, Pandas.

---

## 📊 2. Dataset & Preprocessing

- **Dataset**: `abisee/cnn_dailymail` (version 3.0.0).
- **Initial Training Samples Loaded**: First 10,000 samples.
- **Preprocessing Steps**:
  1. Removed unnecessary metadata (`id` column).
  2. Removed duplicate articles (`subset=['article']`).
  3. Computed word lengths (`article_length` and `highlights_length`).
  4. Filtered dataset ranges:
     - `100 <= article_length <= 1000`
     - `10 <= highlights_length <= 100`
- **Final Cleaned Samples**: 7,957 rows saved locally to `cleaned_data.csv`.
- **Train / Test Split**: 80% Train (6,365 rows) / 20% Test (1,592 rows) with `random_state=42`.

---

## 🤖 3. Model Architecture & Tokenization

- **Original Pretrained Base Model**: `facebook/bart-large-cnn`
- **Fine-Tuned Model**: `omarhosam-237/bart-news-summarizer`
- **Max Input Length**: 1,024 tokens (Byte-Pair Encoding).
- **Max Target Length**: 128 tokens.
- **Decoding Method**: Beam search decoding with `num_beams=4` and `early_stopping=True`.

---

## 🏆 4. Final Fine-Tuning Evaluation Results

These are the exact evaluation metrics achieved in the final successful training epoch:

| Metric | Value |
| :--- | :--- |
| **Epoch** | 1 |
| **Training Loss** | `0.547134` |
| **Validation Loss** | `0.536072` |
| **ROUGE-1** | `41.7358` |
| **ROUGE-2** | `19.8203` |
| **ROUGE-L** | `29.5819` |
| **ROUGE-Lsum** | `39.2892` |

---

## 📁 5. Project Structure

```
Graduation Project/
│
├── app.py                      # Streamlit Home Page & Project Summary
├── utils.py                    # Core utilities (model loading, dataset cache, inference)
├── cleaned_data.csv            # Preprocessed CNN/DailyMail dataset subset (7,957 clean rows)
├── requirements.txt            # Python dependencies for deployment
├── README.md                   # Project documentation
│
└── pages/
    ├── 1_📊_Dataset_Analysis.py # Interactive EDA, filters, and Plotly charts
    └── 2_✍️_Text_Summarization.py# Live text summarization & sample evaluation
```

---

## 💻 6. Local Setup & Execution

### Prerequisites
Python 3.9+ installed on your machine.

### Installation
1. Clone or download this repository.
2. Open terminal in the project root directory and install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Web App
Execute the following command to launch Streamlit:
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## ⚙️ 7. Streamlit App Pages & Features

1. **Home Page (`app.py`)**:
   - High-level overview of abstractive summarization and BART architecture.
   - Live dataset statistics derived from `cleaned_data.csv`.
   - Final evaluation metrics (Loss & ROUGE scores).
2. **Dataset Analysis (`1_📊_Dataset_Analysis.py`)**:
   - Interactive sidebar length sliders.
   - Interactive Plotly visualizations (Histograms, Scatter Plot, Density Heatmap).
   - Filtered metric cards and dataset preview table.
3. **Text Summarization (`2_✍️_Text_Summarization.py`)**:
   - **Custom Article Mode**: Paste any news text, view real-time word/character count, input validation warnings, and generate abstractive summaries.
   - **Sample Article Mode**: Select preprocessed articles from `cleaned_data.csv` to compare **Reference Summary (Human Ground Truth)** vs **Generated Summary (BART Output)**.
   - Compression ratio and length reduction calculation.

---

## 🌐 8. Deployment Limitations & Hosting Considerations

- **Model File Size**: BART-large model weights are ~1.6 GB.
- **Initial Download & Cold Start**: On initial startup, the model downloads from Hugging Face Hub (`omarhosam-237/bart-news-summarizer`). Subsequent runs use Streamlit's `@st.cache_resource`.
- **Memory Requirements**: Minimum 4 GB RAM (8 GB+ recommended) for smooth CPU/GPU inference.
- **Streamlit Community Cloud Limitation**: Free Streamlit Community Cloud provides 1 GB RAM limit, which may result in Out-Of-Memory (OOM) errors during BART-large model loading. For cloud hosting, platforms with higher memory limits (e.g. Hugging Face Spaces with T4 GPU / CPU Basic 16GB RAM, AWS EC2, Render, Docker) are recommended.
