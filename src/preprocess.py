import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def clean_text(text):
    """
    Clean text: lowercase, remove URLs, punctuation, and stopwords.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    
    return " ".join(tokens)

def preprocess_df(df):
    """
    Apply cleaning and numerical label conversion to the dataframe.
    """
    print("\n--- Data Preprocessing ---")
    # Convert labels: spam = 1, ham = 0
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print("Cleaning text messages...")
    df['clean_message'] = df['message'].apply(clean_text)
    print("Preprocessing completed.")
    return df

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    test_df = pd.DataFrame({'label': ['ham', 'spam'], 'message': ['Hello, how are you?', 'WINNER! Click here http://spam.com']})
    print(preprocess_df(test_df))
