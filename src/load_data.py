import pandas as pd
import os

def load_dataset(file_path):
    """
    Load the dataset using pandas and display basic information.
    """
    print(f"Loading dataset from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    # Load with latin-1 encoding as it's common for the SMS Spam dataset
    df = pd.read_csv(file_path, encoding='latin-1')
    
    # Remove unnecessary columns and rename
    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
    df.columns = ['label', 'message']
    
    print("\n--- Dataset Information ---")
    print(df.info())
    print("\n--- Basic Statistics ---")
    print(df.describe())
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    return df

if __name__ == "__main__":
    # Example usage
    data_path = os.path.join("data", "spam.csv")
    load_dataset(data_path)
