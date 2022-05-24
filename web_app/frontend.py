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


title = "Twitter Positivity Analyzer"
description = """
<h2> Description </h2>
Twitter is a social media network on which users post and interact with messages known as "tweets".
It allows an user to post, like, and retweet tweets.

Twitter is also known by the excessive negativity or criticism by a great part of its users.
Considering that, this application intends to classify a tweet according to its positivity.
The positivity is measured in five categories:
- Extremely negative
- Negative
- Neutral
- Positive
- Extremely positive

The application is based on a BERT model fine tuned on this
[dataset](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification).
"""

article = "Check out this \
[repository](https://github.com/hectorLop/Twitter_Positivity_Analyzer) \
with a lot more details about this method and implementation."


app = gr.Interface(
    fn=process_tweet,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Tweet url..."),
    outputs="text",
    title=title,
    description=description,
    article=article,
)

if __name__ == "__main__":
    app, local_url, share_url = app.launch(server_port=8500, server_name="0.0.0.0")
