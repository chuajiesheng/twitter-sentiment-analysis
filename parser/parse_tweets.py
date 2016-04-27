#!/usr/bin/env python3

import code
import logging
import tweet

# Sentiment Analysis - http://www.nltk.org/howto/sentiment.html
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment import util

FORMAT = '[%(asctime)s][%(levelname)-8s] #%(funcName)-10s → %(message)s'
logger = None


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

        return tweet.Tweet(sid, label, text)

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


if __name__ == '__main__':
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    sdb = TweetDatabase()
    tweets = sdb.read_db()
    dataset = sdb.get_tokens(tweets)

    training_tweets_size = int(len(dataset) / 3)
    training_tweets = dataset[:training_tweets_size]
    testing_tweets = dataset[training_tweets_size:]

    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([util.mark_negation(d) for d in training_tweets])
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)

    training_set = sentim_analyzer.apply_features(training_tweets)
    test_set = sentim_analyzer.apply_features(testing_tweets)

    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)

    for key, value in sorted(sentim_analyzer.evaluate(test_set).items()):
        print('{0}: {1}'.format(key, value))

    # code.interact(local=locals())



