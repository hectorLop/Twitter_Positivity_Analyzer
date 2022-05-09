"""
Positivity predictor.

This module provides the functionality to predict
a tweet's positivity using a BERT model.
"""
import torch
from transformers import BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    output_attentions=False,
    output_hidden_states=False,
    local_files_only=True,
)
model.load_state_dict(torch.load("data/BERT_ft_epoch5.model"))
model.eval()


def predict_positivity(text: str) -> str:
    """
    Predict the positivity of a given tweet.

    Args:
        text (str): Tweet's text.

    Returns:
        str: Predicted positivity.
    """
    label_dict = {
        0: "Extremely Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Extremely Positive",
    }
    encoded = tokenizer(text, return_tensors="pt")
    logits = model(**encoded).logits

    predicted_class_id = logits.argmax().item()

    return label_dict[predicted_class_id]
