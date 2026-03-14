import joblib
import os
import pandas as pd
from src.preprocess import clean_text
from src.sentiment_analysis import analyze_sentiment

def predict_spam_classical(message, model_path, vectorizer_path):
    """
    Classical ML (SVM/NB) prediction using TF-IDF.
    """
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Model or Vectorizer not found at {model_path} and {vectorizer_path}")
        
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)
    
    cleaned_message = clean_text(message)
    vectorized_message = tfidf.transform([cleaned_message])
    
    prediction = model.predict(vectorized_message)[0]
    label = "Spam" if prediction == 1 else "Not Spam (Ham)"
    return label, cleaned_message

def predict_spam_bert(message):
    """
    Deep Learning prediction using pre-trained BERT.
    """
    try:
        from src.bert_model import get_bert_prediction
        label, confidence = get_bert_prediction(message)
        return label, confidence
    except Exception as e:
        print(f"Error initializing BERT: {e}")
        return "Error", 0.0

def predict_spam(message, model_path, vectorizer_path, results_dir, method="classical"):
    """
    Predict whether a new message is Spam or Not Spam using specified method and save the result.
    """
    print(f"\n--- Message Prediction ({method.upper()}) ---")
    
    # Ensure results/predictions directory exists
    predictions_dir = os.path.join(results_dir, "predictions")
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
        
    cleaned_message = "N/A"
    confidence = 1.0
    
    if method == "classical":
        label, cleaned_message = predict_spam_classical(message, model_path, vectorizer_path)
    elif method == "bert":
        label, confidence = predict_spam_bert(message)
    else:
        raise ValueError("Method must be 'classical' or 'bert'")
    
    # Perform Sentiment Analysis
    sentiment = analyze_sentiment(message)
    
    # Result Data
    result = {
        'Original Message': message,
        'Method': method,
        'Prediction': label,
        'Confidence/Cleaned': confidence if method == "bert" else cleaned_message,
        'Sentiment': sentiment
    }
    
    print(f"Message: {message}")
    print(f"Prediction: {label}")
    print(f"Sentiment: {sentiment}")
    
    # Save the prediction result
    prediction_path = os.path.join(predictions_dir, f'latest_prediction_{method}.csv')
    result_df = pd.DataFrame([result])
    result_df.to_csv(prediction_path, index=False)
    print(f"Saved prediction result to: {prediction_path}")
    
    return label, sentiment

if __name__ == "__main__":
    # Example usage
    message = "WINNER! You have won a 1000 cash prize. Call 09061701461 to claim now!"
    model_path = os.path.join("models", "spam_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    results_path = "results"
    
    try:
        predict_spam(message, model_path, vectorizer_path, results_path)
    except Exception as e:
        print(f"Error during prediction: {e}")
