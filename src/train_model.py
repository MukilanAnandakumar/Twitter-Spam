from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os

def train_models(X, y, models_dir):
    """
    Split dataset into training and testing sets and train multiple models.
    """
    print("\n--- Model Training ---")
    
    # Ensure models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Split the dataset: 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(probability=True)
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name} Score (Accuracy): {score:.4f}")
        
        trained_models[name] = model
        
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with Accuracy {best_score:.4f}")
    
    # Save the best model
    model_path = os.path.join(models_dir, 'spam_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Saved best model to: {model_path}")
    
    return trained_models, X_test, y_test, best_model_name

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=3000, random_state=42)
    models_path = "models"
    train_models(X, y, models_path)
