# import dataset

FILES = ['./analysis/input/negative_tweets.txt', './analysis/input/neutral_tweets.txt', './analysis/input/positive_tweets.txt']

tweets = []
for file in FILES:
    tweet_set = []
    with open(file, 'r') as f:
        for line in f:
            tweet_set.append(line)

    assert len(tweet_set) == 1367
    tweets.extend(tweet_set)

print('Total number of tweets: {}'.format(len(tweets)))

# import results
import numpy as np
target = np.array([-1] * 1367 + [0] * 1367 + [1] * 1367)

# split train/test 60/40
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tweets, target, test_size=0.1, random_state=1)

print('Train: {},{}'.format(len(X_train), y_train.shape))
print('Test: {},{}'.format(len(X_test), y_test.shape))


# train
class NewTokenizer(object):
    SYMBOLS = [',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', ' ', '&', '$', '-', '|', '«', '`', ' ']

    def __init__(self, n, k):
        from nltk.tokenize.casual import TweetTokenizer
        from nltk.stem import WordNetLemmatizer

        self.tknzr = TweetTokenizer()
        self.wnl = WordNetLemmatizer()

        self.n = n
        self.k = k

    @staticmethod
    def remove_single_symbols(tokens):
        return [t.lower() for t in tokens if t.lower() not in NewTokenizer.SYMBOLS]

    @staticmethod
    def remove_full_stops(tokens):
        return [t.lower() for t in tokens if t.lower() not in [' ', '.']]

    def __call__(self, t):
        from nltk.sentiment.util import mark_negation
        from nltk.util import skipgrams

        tokenised_tweet = self.remove_single_symbols(self.tknzr.tokenize(t))
        lemmatised_tweet = [self.wnl.lemmatize(w) for w in tokenised_tweet]
        three_to_six_chars_words = [w for w in lemmatised_tweet if (2 < len(w) < 7)]
        negated_tweet = mark_negation(three_to_six_chars_words)
        list_of_skipgrams = list(skipgrams(negated_tweet, self.n, self.k))
        features = list([' '.join(self.remove_full_stops(list(s))) for s in list_of_skipgrams])

        return features

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('vect', CountVectorizer(tokenizer=NewTokenizer(2, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='squared_loss', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

pipeline = pipeline.fit(X_train, y_train)

# predict

predicted = pipeline.predict(X_test)
print('Accuracy: {}'.format(np.mean(predicted == y_test)))

X_ones = np.array(X_test)[y_test == 1]
predicted_positive = pipeline.predict(X_ones)
print('Positive accuracy: {}'.format(np.mean(predicted_positive == 1)))

X_ones = np.array(X_test)[y_test == -1]
predicted_negative = pipeline.predict(X_ones)
print('Negative accuracy: {}'.format(np.mean(predicted_negative == -1)))

# metrics

from sklearn import metrics
predicted = pipeline.predict(X_test)
print(metrics.classification_report(y_test, predicted))

print('Confusion matrix: \n{}'.format(metrics.confusion_matrix(y_test, predicted)))

# grid search

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from time import time
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'vect__tokenizer': (NewTokenizer(3, 2), NewTokenizer(2, 2), None),
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__loss': ('squared_loss', 'hinge', 'log', 'epsilon_insensitive'),
    'clf__alpha': (0.0001, 0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__n_iter': (5, 10, 50, 80),
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=8, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=10), verbose=1, scoring='accuracy')

print('Performing grid search...')
print('pipeline: {}'.format([name for name, _ in pipeline.steps]))
print('parameters:')
pprint(parameters)
t0 = time()
grid_search.fit(tweets, target)
print("Done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Best parameters set:
# 	clf__alpha: 1e-05
# 	clf__loss: 'epsilon_insensitive'
# 	clf__n_iter: 80
# 	clf__penalty: 'elasticnet'
# 	tfidf__norm: 'l2'
# 	tfidf__use_idf: False
# 	vect__max_df: 1.0
# 	vect__ngram_range: (1, 2)