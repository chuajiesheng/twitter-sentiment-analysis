#!/usr/bin/env python3

from enum import Enum, unique
from nltk.tokenize.casual import TweetTokenizer
from parser.tokenizer import Tokenizer


class InvalidTweetDataError(ValueError):
    pass


@unique
class Sentiment(Enum):
    negative = -1
    neutral = 0
    positive = 1


class Tweet:
    sid = None
    sentiment = None
    text = None

    def __init__(self, sid, sentiment, text):
        invalid_sid = sid is None or len(sid) < 1
        invalid_text = text is None or len(text) < 1
        invalid_sentiment = sentiment is None or sentiment not in [s.name for s in Sentiment]

        if invalid_sid or invalid_text or invalid_sentiment:
            raise InvalidTweetDataError()

        self.sid = sid
        self.text = text
        self.sentiment = Sentiment[sentiment]

    def get_tokens(self):
        tok = Tokenizer(preserve_case=False)
        return list(tok.tokenize(self.text)), self.sentiment.name
