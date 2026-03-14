import os
from src.load_data import load_dataset
from src.preprocess import preprocess_df
from src.eda_analysis import perform_eda
from src.feature_extraction import extract_features
from src.train_model import train_models
from src.evaluate_model import evaluate_models
from src.sentiment_analysis import perform_sentiment_analysis
from src.predict import predict_spam

def main():
    """
    Main pipeline to run the entire project from start to end.
    """
    print("==========================================================")
    print("Sentiment Analysis to Detect and Prevent Twitter Spam Pipeline")
    print("==========================================================")
    
    # 1. Dataset Loading
    data_path = os.path.join("data", "spam.csv")
    df = load_dataset(data_path)
    
    # 2. Data Preprocessing
    df = preprocess_df(df)
    
    # 3. Exploratory Data Analysis (EDA)
    plots_path = os.path.join("results", "plots")
    perform_eda(df, plots_path)
    
    # 4. Sentiment Analysis
    results_path = "results"
    df = perform_sentiment_analysis(df, results_path)
    
    # 5. Feature Extraction
    models_path = "models"
    X, y, tfidf = extract_features(df, models_path)
    
    # 6. Model Training
    trained_models, X_test, y_test, best_model_name = train_models(X, y, models_path)
    
    # 7. Model Evaluation
    evaluate_models(trained_models, X_test, y_test, results_path)
    
    # 8. Sample Prediction
    sample_message = "URGENT! You have won a 1-week FREE membership to our prize club. Call 08712345678 now!"
    model_path = os.path.join(models_path, 'spam_model.pkl')
    vectorizer_path = os.path.join(models_path, 'tfidf_vectorizer.pkl')
    
    predict_spam(sample_message, model_path, vectorizer_path, results_path)
    
    print("\n==========================================================")
    print("Pipeline Execution Completed Successfully!")
    print("Check the 'results' folder for plots, metrics, and predictions.")
    print("Check the 'models' folder for the trained model and vectorizer.")
    print("==========================================================")

if __name__ == "__main__":
    main()
