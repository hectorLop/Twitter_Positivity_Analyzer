"""
Gradio Twitter analizer application.

This module provides a gradio-based web application
for the Twitter analyzer project.
"""
import json

import boto3
import gradio as gr

from twitter_analyzer.scraper.tweet_scraper import retrieve_tweet_text

# from web_app.backend import predict_positivity


def process_tweet(url: str) -> str:
    """
    Get a tweet's positivity.

    Args:
        url (str): Tweet's URL.

    Returns:
        str: Predicted positivity
    """
    text = retrieve_tweet_text(url)

    payload = {"text": text}

    session = boto3.Session(profile_name="twitter")
    lambda_client = session.client("lambda")
    response = lambda_client.invoke(
        FunctionName="twitter-analyzer-lambda",
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )

    response = json.loads(response["Payload"].read().decode())
    print(response)
    response = json.loads(response["body"])
    outcome = response["label"]

    label_dict = {
        0: "Extremely Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Extremely Positive",
    }

    return label_dict[outcome]


app = gr.Interface(
    fn=process_tweet,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Tweet url..."),
    outputs="text",
)

if __name__ == "__main__":
    app, local_url, share_url = app.launch()
