import numpy as np
from sklearn.metrics import *

def get_dataset():
    files = ['./analysis/input/negative_tweets.txt', './analysis/input/neutral_tweets.txt', './analysis/input/positive_tweets.txt']

    x = []
    for file in files:
        s = []
        with open(file, 'r') as f:
            for line in f:
                s.append(line.strip())

        assert len(s) == 1367
        x.extend(s)

    y = np.array([-1] * 1367 + [0] * 1367 + [1] * 1367)
    return x, y

tweets, target = get_dataset()
# split train/test 60/40

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tweets, target, test_size=0.2, random_state=12)

print('Train: \t\t\tX:{},\tY:{}'.format(len(X_train), y_train.shape[0]))
print('Test: \t\t\tX:{},\tY:{}'.format(len(X_test), y_test.shape[0]))

# train
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

pipeline = Pipeline([('vect', CountVectorizer(max_df=0.75, ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer(norm='l1', use_idf=False)),
                     ('clf', RandomForestClassifier(random_state=0, n_estimators=80, class_weight='auto'))])
pipeline = pipeline.fit(X_train, y_train)

predicted = pipeline.predict(X_test)
print('Accuracy: \t\t{}'.format(accuracy_score(y_test, predicted)))
print('Macro Precision: \t{}'.format(precision_score(y_test, predicted, average='macro')))
print(classification_report(y_test, predicted))

forest = pipeline.steps[2][1]
importances = forest.feature_importances_