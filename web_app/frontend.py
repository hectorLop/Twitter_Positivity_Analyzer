"""
Gradio Twitter analizer application.

This module provides a gradio-based web application
for the Twitter analyzer project.
"""
import json

import boto3
import gradio as gr
from tweet_scraper import retrieve_tweet_text


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

    session = boto3.Session()
    lambda_client = session.client("lambda")
    response = lambda_client.invoke(
        FunctionName="twitter-analyzer-lambda",
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )

    response = json.loads(response["Payload"].read().decode())
    response = json.loads(response["body"])
    outcome = response["label"]

    return outcome


app = gr.Interface(
    fn=process_tweet,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Tweet url..."),
    outputs="text",
)

if __name__ == "__main__":
    app, local_url, share_url = app.launch(server_port=8500, server_name="0.0.0.0")
