import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import joblib

# Define the path to the saved model and label encoder
models_dir = os.path.join(os.getcwd(), 'saved_model')

# Load the label encoder and model
label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
model = BertForSequenceClassification.from_pretrained('saved_model')
tokenizer = BertTokenizer.from_pretrained('saved_model')

# Use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


def prediction_pipe(text: str) -> str:
    """
    Predicts the department label for the given email text using a pre-trained BERT model.

    Args:
        text (str): The email text or content to be classified.

    Returns:
        str: The predicted department label.
    """
    # Set the model to evaluation mode
    model.eval()

    # Tokenize the input text and move tensors to the selected device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Disable gradient calculations (inference mode)
    with torch.inference_mode():
        # Get model outputs
        outputs = model(**inputs)

        # Extract logits (raw model predictions)
        logits = outputs.logits

        # Apply softmax to convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get the index of the highest probability class
        predicted_class_idx = torch.argmax(probs, dim=-1).item()

        # Convert the predicted index to the corresponding label using the label encoder
        predicted_class = label_encoder.inverse_transform([predicted_class_idx]).item()

    return predicted_class
