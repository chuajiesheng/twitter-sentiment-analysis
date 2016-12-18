# import dataset

import json
INPUT_FILE = './analysis/input/dev_posts.json'

tweets = []
with open(INPUT_FILE, 'r') as f:
    for line in f:
        t = json.loads(line)
        tweets.append(t['body'])

print('Total number of tweets: {}'.format(len(tweets)))

# import results

import numpy as np
TARGET_FILE = './analysis/input/test_results.csv'

f = open(TARGET_FILE)
target = np.loadtxt(f)

# split train/test 60/40

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tweets, target, test_size=0.4, random_state=1)

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
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

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
