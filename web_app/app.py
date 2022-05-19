"""Lambda function code."""
import json
from typing import Dict

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
model.load_state_dict(torch.load("model_dir/BERT_ft_epoch5.model"))
model.eval()


def format_response(body: Dict, status_code: int) -> Dict:
    """
    Add format to the lambda response.

    Args:
        body (Dict): Response body.
        status_code (int): Response status.

    Returns:
        Dict: Response format.
    """
    return {
        "statusCode": str(status_code),
        "body": json.dumps(body),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }


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


def handler(event, context):
    """Lambda hanlder function."""
    try:
        text = event["text"]

        pred = predict_positivity(text)

        payload = {"label": pred}

        return format_response(payload, 200)
    except Exception:
        return format_response({"msg": "ERROR"}, 200)
