import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

def perform_eda(df, plots_dir):
    """
    Perform exploratory data analysis and save plots to the results/plots folder.
    """
    print("\n--- Exploratory Data Analysis ---")
    
    # Ensure plots directory exists
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # Label Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Distribution of Spam vs Ham')
    plt.savefig(os.path.join(plots_dir, 'label_distribution.png'))
    plt.close()
    
    print(f"Spam count: {df[df['label'] == 'spam'].shape[0]}")
    print(f"Ham count: {df[df['label'] == 'ham'].shape[0]}")

    # Word Cloud for Spam
    spam_words = " ".join(df[df['label'] == 'spam']['clean_message'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Spam Messages')
    plt.savefig(os.path.join(plots_dir, 'spam_wordcloud.png'))
    plt.close()
    
    print(f"Saved EDA plots to: {plots_dir}")

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    test_df = pd.DataFrame({'label': ['ham', 'spam', 'ham'], 'clean_message': ['hello friend', 'win cash free', 'see you soon']})
    plots_path = os.path.join("results", "plots")
    perform_eda(test_df, plots_path)
