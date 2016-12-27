# import dataset

FILES = ['./analysis/input/negative_tweets.txt', './analysis/input/neutral_tweets.txt', './analysis/input/positive_tweets.txt']

tweets = []
for file in FILES:
    tweet_set = []
    with open(file, 'r') as f:
        for line in f:
            tweet_set.append(line.strip())

    assert len(tweet_set) == 1367
    tweets.extend(tweet_set)

print('Total number of tweets: {}'.format(len(tweets)))

# import results
import numpy as np
target = np.array([-1] * 1367 + [0] * 1367 + [1] * 1367)

# split train/test 60/40

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tweets, target, test_size=0.2, random_state=12)

print('Train: {},{}'.format(len(X_train), y_train.shape))
print('Test: {},{}'.format(len(X_test), y_test.shape))

# train

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('vect', CountVectorizer(max_df=0.75, ngram_range=(1,2))),
                     ('tfidf', TfidfTransformer(norm='l1', use_idf=False)),
                     ('clf', ExtraTreesClassifier(random_state=0, n_estimators=10, class_weight='auto'))])

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

confusion_matrix = metrics.confusion_matrix(y_test, predicted)
normalised_confusion_matrix = confusion_matrix / confusion_matrix.astype(np.float).sum(axis=1)
print('Confusion matrix: \n{}'.format(confusion_matrix))
print('Normalised Confusion matrix: \n{}'.format(normalised_confusion_matrix))

from sklearn import metrics
print('Precision: \t{}'.format(metrics.precision_score(y_test, predicted, average=None)))
print('Recall: \t{}'.format(metrics.recall_score(y_test, predicted, average=None)))
print('F1: \t\t{}'.format(metrics.f1_score(y_test, predicted, average=None)))

print('Macro Precision: \t{}'.format(metrics.precision_score(y_test, predicted, average='macro')))
print('Macro Recall: \t\t{}'.format(metrics.recall_score(y_test, predicted, average='macro')))
print('Macro F1: \t\t{}'.format(metrics.f1_score(y_test, predicted, average='macro')))
print('--------------------------------------------------------------------------------')

# stratified k-fold

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=15)
for train, test in skf.split(tweets, target):
    X_train = np.array(tweets)[train]
    y_train = target[train]

    X_test = np.array(tweets)[test]
    y_test = target[test]

    pipeline = Pipeline([('vect', CountVectorizer(max_df=0.75, ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer(norm='l1', use_idf=False)),
                         ('clf', ExtraTreesClassifier(random_state=0, n_estimators=10, class_weight='auto'))])
    pipeline = pipeline.fit(X_train, y_train)

    predicted = pipeline.predict(X_test)
    print('Accuracy: {}'.format(np.mean(predicted == y_test)))
    print(metrics.classification_report(y_test, predicted))
    print('Macro Precision: \t{}'.format(metrics.precision_score(y_test, predicted, average='macro')))
    print('Macro Recall: \t\t{}'.format(metrics.recall_score(y_test, predicted, average='macro')))
    print('Macro F1: \t\t{}'.format(metrics.f1_score(y_test, predicted, average='macro')))
    print('.')


print('--------------------------------------------------------------------------------')

# grid search

from sklearn.model_selection import GridSearchCV
from pprint import pprint
from time import time
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__n_estimators': (10, 20, 30),
    'clf__random_state': (0, 1),
    'clf__class_weight': ('auto', 'balanced', {-1: 0.5, 0: 0.01, 1: 0.49})
}
scores = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
for score in scores:
    print('Scoring: {}'.format(score))
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5, scoring=score)

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

    print('--------------------------------------------------------------------------------')
