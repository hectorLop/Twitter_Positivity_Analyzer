"""
Twitter scraper.

This module provides the functionality to retrieve
a tweet's text given a tweet's URL.
"""
import re

import requests


def retrieve_tweet_text(tweet_url: str) -> str:
    """
    Retrieve a tweet's text.

    Args:
        tweet_url (url): Tweet's URL.

    Returns:
        str: Tweet's parsed text.
    """
    # Get the url to retrieve tweet-related data
    url = (
        "https://publish.twitter.com/oembed?dnt=true",
        f"&omit_script=true&url={tweet_url}",
    )
    url = str.join("", url)

    # Get the raw html containing th tweet text
    raw_html = requests.get(url).json()["html"]
    # Remove links from text
    pattern = r"<[a][^>]*>(.+?)</[a]>"
    html = re.sub(pattern, "", raw_html)

    # Remove the HTML tags from the text
    text = [i.strip() for i in re.sub("<.*?>", "", html).splitlines() if i][0]

    # If there is a picture, remove all the text after it
    if "pic" in text:
        idx = text.index("pic")
        text = text[:idx]
    # If there is no picture, the &mdash defines the tweet's
    # end.
    elif "&mdash" in text:
        idx = text.index("&mdash")
        text = text[:idx]

    return text
