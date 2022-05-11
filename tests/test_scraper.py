"""
Data scraper unitary testing.

This module contains the unitary testing
for the tweets data scraper.
"""
from twitter_analyzer.scraper.tweet_scraper import retrieve_tweet_text


def test_retrieve_tweet_text():
    """Test the retrieve_tweet_text function."""
    test_url = "https://twitter.com/hectorLop_/status/1493475841526972420"

    text = retrieve_tweet_text(test_url)
    expected_text = "I have tried it, and it works pretty well!"

    assert text == expected_text
