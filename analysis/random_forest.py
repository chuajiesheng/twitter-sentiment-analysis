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
print('------------------------------- Best Parameters --------------------------------')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('vect', CountVectorizer(max_df=0.75, ngram_range=(1,2))),
                     ('tfidf', TfidfTransformer(norm='l1', use_idf=False)),
                     ('clf', RandomForestClassifier(random_state=0, n_estimators=80, class_weight='auto'))])

pipeline = pipeline.fit(X_train, y_train)

# predict

predicted = pipeline.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy: {}'.format(accuracy_score(y_test, predicted)))

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

import code
code.interact(local=dict(globals(), **locals()))