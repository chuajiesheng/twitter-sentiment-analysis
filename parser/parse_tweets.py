#!/usr/bin/env python3

import code
import tweet


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
        pair = line.split('\t')
        if len(pair) != 2: raise InvalidFormatError()
        return tweet.Tweet(pair[0], pair[1])

    def read_db(self):
        with open(self.db_path) as f:
            lines = [line.rstrip('\n') for line in f]
        tweets_arr = list(map(lambda l: TweetDatabase.parse_line(l), lines))
        tweets_dict = {tweet.sid: tweet for tweet in tweets_arr}
        return tweets_dict


class SentimentDatabase(TweetDatabase):

    TASK_A_DEV_DB = 'data/semeval2016/dev/100_topics_100_tweets.sentence-three-point.subtask-A.dev.gold.txt'
    dev_db_path = TASK_A_DEV_DB

    def __init__(self, dev_db=TASK_A_DEV_DB, db_path=TweetDatabase.TWEET_DB):
        TweetDatabase.__init__(self, db_path=db_path)
        self.dev_db_path = dev_db

    @staticmethod
    def parse_labelled_tweet(line):
        if line is None:
            raise InvalidFormatError()

        pair = line.split('\t')

        if len(pair) != 2:
            raise InvalidFormatError()

        sid = pair[0]
        sentiment = pair[1]
        return {'sid': sid, 'sentiment': sentiment}

    def read_dev_db(self):
        with open(self.dev_db_path) as f:
            lines = [line.rstrip('\n') for line in f]
        dicts = list(map(lambda l: SentimentDatabase.parse_labelled_tweet(l), lines))
        tweet_labels = {label['sid']: label['sentiment'] for label in dicts}
        return tweet_labels

    @staticmethod
    def get_labelled_tweets(tweets, labels):
        missing_label = 0
        list_of_labelled_tweets = []
        for t in tweets:
            if t.sid not in labels.keys():
                missing_label += 1
                continue

            list_of_labelled_tweets.append(tweet.SentimentTweet(t, labels[t.sid]))

        print('Total missing labels: ', missing_label)
        return list_of_labelled_tweets


if __name__ == '__main__':
    tweet_db = SentimentDatabase()
    tweets = tweet_db.read_db()
    labels = tweet_db.read_dev_db()
    labelled_tweets = tweet_db.get_labelled_tweets(tweets, labels)
    code.interact(local=locals())



