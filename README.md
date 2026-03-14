# Sentiment Analysis to Detect and Prevent Twitter Spam

A comprehensive machine learning and deep learning project for detecting spam in text messages (SMS/Twitter) while simultaneously performing sentiment analysis.

## 🚀 Overview
This project implements a modular machine learning pipeline that classifies messages as **Spam** or **Ham** (Not Spam). It features:
- **Classical ML Models**: Naive Bayes, Logistic Regression, and SVM with TF-IDF N-gram vectorization.
- **Deep Learning**: Integration with a pre-trained BERT model for state-of-the-art accuracy.
- **Sentiment Analysis**: Real-time sentiment classification (Positive, Neutral, Negative) using TextBlob.
- **Interactive Web App**: A Streamlit-based dashboard for real-time testing and visualization.

## 📂 Project Structure
```text
Twitter Spam/
├── data/               # Raw dataset (spam.csv)
├── src/                # Modular source code
│   ├── load_data.py    # Data loading
│   ├── preprocess.py   # Text cleaning
│   ├── eda_analysis.py # Visualizations
│   ├── feature_extraction.py # TF-IDF N-grams
│   ├── train_model.py  # Model training
│   ├── evaluate_model.py # Metrics/Plots
│   ├── sentiment_analysis.py # Sentiment logic
│   ├── bert_model.py   # Deep Learning module
│   └── predict.py      # Reusable prediction logic
├── models/             # Saved .pkl model files
├── results/            # Saved plots, metrics, and predictions
├── main.py             # Pipeline orchestrator
└── app.py              # Streamlit Web Application
```

## 🛠️ Installation & Setup

1. **Clone or Extract** the project folder.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the Models**:
   Run the full pipeline to generate models and plots:
   ```bash
   python main.py
   ```

## 🖥️ Usage

### Running the Web App
To launch the interactive dashboard:
```bash
streamlit run app.py
```

### Making Predictions via CLI
You can also run predictions for single messages:
```bash
python src/predict.py
```

## 📊 Results
- All plots (Confusion Matrices, WordClouds) are saved in `results/plots/`.
- Detailed model performance metrics are saved in `results/metrics/model_metrics.csv`.
- Sentiment summaries are saved in `results/metrics/sentiment_summary.csv`.

## 🤖 Models Used
- **Classical**: Multinomial Naive Bayes, Logistic Regression, Support Vector Machine (SVM).
- **Deep Learning**: BERT (Bidirectional Encoder Representations from Transformers) via Hugging Face.

---
Developed as a complete Spam Detection and Sentiment Analysis Pipeline.
