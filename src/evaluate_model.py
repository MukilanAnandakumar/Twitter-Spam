from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def evaluate_models(trained_models, X_test, y_test, results_dir):
    """
    Evaluate models and save metrics and confusion matrix plots.
    """
    print("\n--- Model Evaluation ---")
    
    plots_dir = os.path.join(results_dir, "plots")
    metrics_dir = os.path.join(results_dir, "metrics")
    
    # Ensure directories exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
        
    all_metrics = []
    
    for name, model in trained_models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Results for {name}:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        all_metrics.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        # Save Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(plots_dir, f'cm_{name.replace(" ", "_").lower()}.png')
        plt.savefig(cm_path)
        plt.close()
        
    # Save all metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(metrics_dir, 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix plots to: {plots_dir}")
    
    return metrics_df

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.naive_bayes import MultinomialNB
    X, y = make_classification(n_samples=100, n_features=3000, random_state=42)
    # Mocking trained models
    trained_models = {"Naive Bayes": MultinomialNB().fit(X, y)}
    results_path = "results"
    evaluate_models(trained_models, X, y, results_path)
