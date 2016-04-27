#!/usr/bin/env python3

import logging
from parser.tweet import Tweet


class InvalidFormatError(ValueError):
    pass


class TweetDatabase:

    TWEET_DB = 'data/semeval2016/tweets.db'
    db_path = TWEET_DB

    def __init__(self, db_path=TWEET_DB):
        self.db_path = db_path

    @staticmethod
    def parse_line(line):
        if line is None:
            raise InvalidFormatError()

        pair = line.split('\t')

        if len(pair) < 3:
            raise InvalidFormatError()

        sid = pair[0]
        label = pair[1]
        text = pair[2]

        return Tweet(sid, label, text)

    def read_db(self):
        with open(self.db_path) as f:
            lines = [line.rstrip('\n') for line in f]
        tweets_arr = list(map(lambda l: TweetDatabase.parse_line(l), lines))
        logging.info('Total records: %d', len(tweets_arr))

        tweets_dict = {tweet.sid: tweet for tweet in tweets_arr}
        logging.info('Total unique records: %d', len(tweets_dict.items()))
        return tweets_dict

    def get_tokens(self, tweets):
        return [tweets[k].get_tokens() for k in tweets.keys()]
