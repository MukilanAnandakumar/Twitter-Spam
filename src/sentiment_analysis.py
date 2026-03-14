from textblob import TextBlob
import os
import pandas as pd

def analyze_sentiment(text):
    """
    Determine the sentiment of each message: Positive, Negative, or Neutral.
    """
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def perform_sentiment_analysis(df, results_dir):
    """
    Perform sentiment analysis on the entire dataset and save results.
    """
    print("\n--- Sentiment Analysis ---")
    
    # Ensure results directory exists
    metrics_dir = os.path.join(results_dir, "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
        
    # Perform sentiment analysis
    df['sentiment'] = df['message'].apply(analyze_sentiment)
    
    sentiment_counts = df['sentiment'].value_counts()
    print("Sentiment Analysis completed.")
    print(sentiment_counts)
    
    # Save sentiment summary
    sentiment_summary_path = os.path.join(metrics_dir, 'sentiment_summary.csv')
    sentiment_counts.to_csv(sentiment_summary_path)
    print(f"Saved sentiment summary to: {sentiment_summary_path}")
    
    return df

if __name__ == "__main__":
    # Example usage
    test_df = pd.DataFrame({'message': ['I love this!', 'I hate this.', 'Hello there.']})
    results_path = "results"
    perform_sentiment_analysis(test_df, results_path)
