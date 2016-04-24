#!/usr/bin/env python3

import code
from tweet import Tweet


class InvalidFormatError(ValueError):
    pass


class TweetDatabase:

    TWEET_DB = 'data/semeval2016/tweets.db'
    db_path = TWEET_DB

    def __init__(self, db_path=TWEET_DB):
        self.db_path = db_path

    @staticmethod
    def parse_line(line):
        if line is None: raise InvalidFormatError()

        assert line is not None
        pair = line.split('\t')
        if len(pair) != 2: raise InvalidFormatError()

        return Tweet(pair[0], pair[1])

    def read_db(self):
        with open(self.db_path) as f:
            lines = [line.rstrip('\n') for line in f]
        pairs = list(map(lambda l: TweetDatabase.parse_line(l), lines))
        return pairs

if __name__ == '__main__':
    tweet_db = TweetDatabase()
    pairs = tweet_db.read_db()
    code.interact(local=locals())


