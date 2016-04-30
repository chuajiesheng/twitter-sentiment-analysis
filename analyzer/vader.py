#!/usr/bin/env python3

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from base_classifier import *
from parser import *

logger = None

CLASSIFIER_NAME = 'vader'

class VaderClassifier(Classifier):

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

        return correct, wrong

if __name__ == '__main__':
    print('------------- Vader Classifier -------------')
    vc = VaderClassifier()
    vc.init_logging()
    correct, wrong = vc.test()

    with open(Classifier.get_output_file(CLASSIFIER_NAME), 'w') as output_file:
        output = 'Total correct: {}\n'.format(correct)
        output += 'Total wrong: {}\n'.format(wrong)
        output += 'Total accuracy: {}\n'.format(float(correct) / (correct + wrong))

        print(output)
        output_file.write(output)

    # code.interact(local=locals())
