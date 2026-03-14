from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def extract_features(df, models_dir):
    """
    Convert text into numerical features using TF-IDF vectorization.
    """
    print("\n--- Feature Extraction (TF-IDF) ---")
    
    # Ensure models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Using unigrams and bigrams (1, 2) to capture more context
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['clean_message'])
    y = df['label_num']
    
    # Save the vectorizer for prediction
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    joblib.dump(tfidf, vectorizer_path)
    print(f"Saved TF-IDF vectorizer to: {vectorizer_path}")
    
    return X, y, tfidf

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    test_df = pd.DataFrame({'clean_message': ['hello friend', 'win cash free'], 'label_num': [0, 1]})
    models_path = "models"
    X, y, tfidf = extract_features(test_df, models_path)
    print(f"Feature matrix shape: {X.shape}")
