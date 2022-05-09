"""
Gradio Twitter analizer application.

This module provides a gradio-based web application
for the Twitter analyzer project.
"""
import gradio as gr

from twitter_analyzer.scraper.tweet_scraper import retrieve_tweet_text
from web_app.backend import predict_positivity


def process_tweet(url: str) -> str:
    """
    Get a tweet's positivity.

    Args:
        url (str): Tweet's URL.

    Returns:
        str: Predicted positivity
    """
    text = retrieve_tweet_text(url)
    outcome = predict_positivity(text)

    return outcome


app = gr.Interface(
    fn=process_tweet,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Tweet url..."),
    outputs="text",
)

if __name__ == "__main__":
    app, local_url, share_url = app.launch()
