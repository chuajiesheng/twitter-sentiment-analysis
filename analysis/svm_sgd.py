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
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

text_clf = text_clf.fit(X_train, y_train)

# predict

predicted = text_clf.predict(X_test)
print('Accuracy: {}'.format(np.mean(predicted == y_test)))

X_ones = np.array(X_test)[y_test == 1]
predicted_positive = text_clf.predict(X_ones)
print('Positive accuracy: {}'.format(np.mean(predicted_positive == 1)))

X_ones = np.array(X_test)[y_test == -1]
predicted_negative = text_clf.predict(X_ones)
print('Negative accuracy: {}'.format(np.mean(predicted_negative == -1)))

# metrics

from sklearn import metrics
predicted = text_clf.predict(X_test)
print(metrics.classification_report(y_test, predicted))

print('Confusion matrix: \n{}'.format(metrics.confusion_matrix(y_test, predicted)))
