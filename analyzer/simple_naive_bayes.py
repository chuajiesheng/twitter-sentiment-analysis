#!/usr/bin/env python3

# Sentiment Analysis - http://www.nltk.org/howto/sentiment.html
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment import util

from analyzer import Classifier
from parser import TweetDatabase

CLASSIFIER_NAME = 'simple_naive_bayes'


class SimpleNaiveBayesClassifier(Classifier):

    dataset = None

    def __init__(self):
        pass

    def get_dataset(self):
        sdb = TweetDatabase()
        tweets = sdb.read_db()
        return sdb.get_tokens(tweets)

    def split_dataset(self, dataset):
        training_size = int(len(dataset) / 2)
        training = dataset[:training_size]
        testing = dataset[training_size:]
        return training, testing

    def train(self):
        snb = SimpleNaiveBayesClassifier()
        self.dataset = snb.get_dataset()
        training_tweets, _ = snb.split_dataset(self.dataset)

        sentim_analyzer = SentimentAnalyzer()
        all_words_neg = sentim_analyzer.all_words([util.mark_negation(d) for d in training_tweets])
        unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
        bigram_feats = sentim_analyzer.bigram_collocation_feats(all_words_neg)
        sentim_analyzer.add_feat_extractor(util.extract_unigram_feats, unigrams=unigram_feats)
        sentim_analyzer.add_feat_extractor(util.extract_bigram_feats, bigrams=bigram_feats)

        training_set = sentim_analyzer.apply_features(training_tweets)

        trainer = NaiveBayesClassifier.train
        classifier = sentim_analyzer.train(trainer, training_set)

        return sentim_analyzer

    def test(self, analyzer):
        _, testing_tweets = snb.split_dataset(self.dataset)
        test_set = analyzer.apply_features(testing_tweets)

        return sorted(analyzer.evaluate(test_set).items())

if __name__ == '__main__':
    print('------ Simple Naive Bayes Classifier -------')
    snb = SimpleNaiveBayesClassifier()
    snb.init_logging()
    analyzer = snb.train()
    result = snb.test(analyzer)

    with open(Classifier.get_output_file(CLASSIFIER_NAME), 'w') as output_file:
        for key, value in result:
            output = '{0}: {1}'.format(key, value)
            print(output)
            output_file.write(output)

    # code.interact(local=locals())
