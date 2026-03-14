from transformers import pipeline
import torch
import os

class BERTSpamDetector:
    def __init__(self, model_name="mrm8488/bert-tiny-finetuned-sms-spam-detection"):
        """
        Initialize the BERT-based spam detector using a pre-trained model from Hugging Face.
        """
        print(f"Initializing BERT model: {model_name}...")
        # Use CPU if CUDA is not available
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline("text-classification", model=model_name, device=device)

    def predict(self, message):
        """
        Predict whether a message is spam or ham using BERT.
        """
        results = self.classifier(message)
        # The model usually returns labels like 'LABEL_0' (ham) and 'LABEL_1' (spam) 
        # or 'ham'/'spam' depending on how it was trained.
        # For this specific model, it returns 'LABEL_0' for ham and 'LABEL_1' for spam.
        
        prediction = results[0]['label']
        confidence = results[0]['score']
        
        # Mapping labels to human-readable format
        # Note: Check the specific model's config if needed, but for mrm8488's model:
        # LABEL_0 -> ham, LABEL_1 -> spam
        label = "Spam" if prediction == "LABEL_1" or prediction.lower() == "spam" else "Not Spam (Ham)"
        
        return label, confidence

def get_bert_prediction(message):
    """
    Helper function to get BERT prediction for a single message.
    """
    detector = BERTSpamDetector()
    return detector.predict(message)

if __name__ == "__main__":
    # Example usage
    sample_msg = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim."
    label, conf = get_bert_prediction(sample_msg)
    print(f"Message: {sample_msg}")
    print(f"BERT Prediction: {label} (Confidence: {conf:.4f})")
