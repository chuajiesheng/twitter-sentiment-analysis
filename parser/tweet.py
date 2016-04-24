#!/usr/bin/env python3
from enum import Enum, unique


class Tweet:
    sid = None
    text = None

    def __init__(self, sid, text):
        self.sid = sid
        self.text = text


@unique
class Sentiment(Enum):
    negative = -1
    neutral = 0
    positive = 1


class SentimentTweet(Tweet):
    sentiment = Sentiment.neutral

    def __init__(self, tweet, sentiment):
        assert tweet is not None
        assert sentiment is not None
        assert sentiment in [s.name for s in Sentiment]

        Tweet.__init__(self, tweet.sid, tweet.text)
        self.sentiment = Sentiment[sentiment]

