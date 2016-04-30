#!/usr/bin/env python3

import logging

# Sentiment Analysis - http://www.nltk.org/howto/sentiment.html
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from parser import *

FORMAT = '[%(asctime)s][%(levelname)-8s] #%(funcName)-10s → %(message)s'

logger = None


class VaderClassifier:

    def __init__(self):
        pass


    def get_tweets(self):
        sdb = TweetDatabase()
        tweets = sdb.read_db()
        return tweets

    @staticmethod
    def get_highest_possibility(polarity_scores):
        neg = polarity_scores['neg']
        neu = polarity_scores['neu']
        pos = polarity_scores['pos']

        if neg > neu and neg > pos:
            return Sentiment.negative
        elif neu > neg and neu > pos:
            return Sentiment.neutral
        else:
            assert pos > neg and pos >= neu
            return Sentiment.positive

    def test(self):
        correct = 0
        wrong = 0

        sid = SentimentIntensityAnalyzer()

        dataset = self.get_tweets()
        for k in dataset.keys():
            tweet = dataset[k]
            ss = sid.polarity_scores(tweet.text)
            sentiment = self.get_highest_possibility(ss)

            if tweet.sentiment == sentiment:
                correct += 1
            else:
                wrong += 1

        logging.info('Total correct: %d', correct)
        logging.info('Total wrong: %d', wrong)
        logging.info('Total accuracy: %f', float(correct) / (correct + wrong))

if __name__ == '__main__':
    print('------------- Vader Classifier -------------')
    vc = VaderClassifier()
    vc.init_logging()
    result = vc.test()

    # code.interact(local=locals())
