import streamlit as st
import joblib
import os
import sys
from src.preprocess import clean_text
from src.sentiment_analysis import analyze_sentiment

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_classical_resources():
    model_path = os.path.join('models', 'spam_model.pkl')
    tfidf_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        return None, None
        
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    return model, tfidf

@st.cache_resource
def load_bert_detector():
    try:
        from src.bert_model import BERTSpamDetector
        return BERTSpamDetector()
    except Exception as e:
        st.error(f"Failed to load BERT model: {e}")
        return None

# --- Main App ---
def main():
    st.set_page_config(page_title="Twitter Spam & Sentiment Detector", layout="wide")
    
    st.title("🛡️ Twitter Spam & Sentiment Detector")
    st.markdown("""
    This app predicts whether a message is **Spam** or **Ham** (Not Spam) and also determines its **Sentiment**.
    Choose between **Classical ML (SVM/NB)** and **Deep Learning (BERT)**.
    """)
    
    # Sidebar for Model Selection and Insights
    st.sidebar.title("⚙️ Model Settings")
    model_choice = st.sidebar.radio("Select Detection Model:", ("Classical ML (SVM)", "Deep Learning (BERT)"))
    
    st.sidebar.title("📊 Project Insights")
    plots_dir = os.path.join('results', 'plots')
    
    if st.sidebar.checkbox("Show Dataset Distribution"):
        dist_plot = os.path.join(plots_dir, 'label_distribution.png')
        if os.path.exists(dist_plot):
            st.sidebar.image(dist_plot, caption="Spam vs Ham Count")
        else:
            st.sidebar.warning("Run `python main.py` first to generate plots.")
            
    if st.sidebar.checkbox("Show Spam WordCloud"):
        wc_plot = os.path.join(plots_dir, 'spam_wordcloud.png')
        if os.path.exists(wc_plot):
            st.sidebar.image(wc_plot, caption="Common Words in Spam")
            
    if st.sidebar.checkbox("Show Best Model Performance"):
        metrics_path = os.path.join('results', 'metrics', 'model_metrics.csv')
        if os.path.exists(metrics_path):
            import pandas as pd
            metrics_df = pd.read_csv(metrics_path)
            st.sidebar.dataframe(metrics_df)
        else:
            st.sidebar.warning("Run `python main.py` first to see metrics.")

    # Main Input Area
    st.subheader("📝 Enter Message")
    user_input = st.text_area("Paste the message you want to check here:", height=150)
    
    if st.button("Predict & Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            try:
                prediction = None
                confidence = None
                cleaned_text = "N/A"
                
                if model_choice == "Classical ML (SVM)":
                    model, tfidf = load_classical_resources()
                    if model is None or tfidf is None:
                        st.error("Classical model files not found. Run `python main.py` first.")
                    else:
                        cleaned_text = clean_text(user_input)
                        vectorized_text = tfidf.transform([cleaned_text])
                        pred_val = model.predict(vectorized_text)[0]
                        prediction = "Spam" if pred_val == 1 else "Not Spam (Ham)"
                        # Get probability if available
                        if hasattr(model, "predict_proba"):
                            confidence = max(model.predict_proba(vectorized_text)[0])
                
                else: # Deep Learning (BERT)
                    with st.spinner("BERT is analyzing..."):
                        detector = load_bert_detector()
                        if detector:
                            prediction, confidence = detector.predict(user_input)
                        else:
                            st.error("BERT model failed to load.")

                if prediction:
                    # Sentiment Analysis
                    sentiment = analyze_sentiment(user_input)
                    
                    # Display Results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"🔍 {model_choice} Prediction")
                        if prediction == "Spam":
                            st.error(f"🚨 **SPAM** Detected!")
                        else:
                            st.success(f"✅ **{prediction}**")
                        
                        if confidence is not None:
                            st.write(f"**Confidence:** {confidence:.2f}")
                    
                    with col2:
                        st.subheader("😊 Sentiment Analysis")
                        if sentiment == 'Positive':
                            st.info(f"✨ Sentiment: **{sentiment}**")
                        elif sentiment == 'Negative':
                            st.error(f"🛑 Sentiment: **{sentiment}**")
                        else:
                            st.warning(f"😐 Sentiment: **{sentiment}**")
                    
                    # Details
                    with st.expander("Details"):
                        if model_choice == "Classical ML (SVM)":
                            st.write(f"**Preprocessed Text:** *'{cleaned_text}'*")
                        st.write(f"**Method Used:** {model_choice}")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.markdown("---")
    st.caption("Twitter Spam Detection System - Modular ML Project")

if __name__ == "__main__":
    main()
