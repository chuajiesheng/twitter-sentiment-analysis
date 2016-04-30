#!/usr/bin/env python3

import logging
import os

# Sentiment Analysis - http://www.nltk.org/howto/sentiment.html
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment import util

from parser import TweetDatabase

FORMAT = '[%(asctime)s][%(levelname)-8s] #%(funcName)-10s → %(message)s'
OUTPUT_FILE = 'result/{0}/{1}.out'
CLASSIFIER = 'simple_naive_bayes'

logger = None


class SimpleNaiveBayesClassifier:

    def __init__(self):
        pass

    def init_logging(self):
        logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    def get_dataset(self):
        sdb = TweetDatabase()
        tweets = sdb.read_db()
        return sdb.get_tokens(tweets)

    def split_dataset(self, dataset):
        training_size = int(len(dataset) / 3)
        training = dataset[:training_size]
        testing = dataset[training_size:]
        return training, testing

    def train(self):
        snb = SimpleNaiveBayesClassifier()
        dataset = snb.get_dataset()
        training_tweets, testing_tweets = snb.split_dataset(dataset)

        sentim_analyzer = SentimentAnalyzer()
        all_words_neg = sentim_analyzer.all_words([util.mark_negation(d) for d in training_tweets])
        unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
        sentim_analyzer.add_feat_extractor(util.extract_unigram_feats, unigrams=unigram_feats)

        training_set = sentim_analyzer.apply_features(training_tweets)
        test_set = sentim_analyzer.apply_features(testing_tweets)

        trainer = NaiveBayesClassifier.train
        classifier = sentim_analyzer.train(trainer, training_set)

        return sorted(sentim_analyzer.evaluate(test_set).items())

if __name__ == '__main__':
    print('------ Simple Naive Bayes Classifier -------')
    snb = SimpleNaiveBayesClassifier()
    snb.init_logging()
    result = snb.train()

    counter = os.environ['SNAP_PIPELINE_COUNTER']
    with open(OUTPUT_FILE.format(counter, CLASSIFIER), 'w') as output_file:
        for key, value in result:
            output = '{0}: {1}'.format(key, value)
            print(output)
            output_file.write(output)

    # code.interact(local=locals())
