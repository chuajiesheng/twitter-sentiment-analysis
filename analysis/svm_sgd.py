from sklearn.feature_extraction.text import *
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import *
from sklearn.metrics import *
from tokenizers import SkipgramTokenizer
import numpy as np
from pprint import pprint
from time import time


# import dataset
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
X_train, X_test, y_train, y_test = train_test_split(tweets, target, test_size=0.1, random_state=1)
print('Train: \t{},{}'.format(len(X_train), y_train.shape))
print('Test: \t{},{}'.format(len(X_test), y_test.shape))

pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                     ('tfidf', TfidfTransformer(norm='l2', use_idf=True)),
                     ('clf', SGDClassifier(loss='squared_loss', penalty='l2', alpha=1e-04, n_iter=50, random_state=42))])

pipeline = pipeline.fit(X_train, y_train)

# predict
predicted = pipeline.predict(X_test)
print('Accuracy: \t\t{}'.format(accuracy_score(y_test, predicted)))
print('Macro F1: \t\t{}'.format(f1_score(y_test, predicted, average='macro')))

X_ones = np.array(X_test)[y_test == 1]
predicted_positive = pipeline.predict(X_ones)
print('Positive accuracy: \t{}'.format(np.mean(predicted_positive == 1)))

X_ones = np.array(X_test)[y_test == -1]
predicted_negative = pipeline.predict(X_ones)
print('Negative accuracy: \t{}'.format(np.mean(predicted_negative == -1)))

# metrics
predicted = pipeline.predict(X_test)
print(classification_report(y_test, predicted))
print('Confusion matrix: \n{}'.format(confusion_matrix(y_test, predicted)))

# grid search
parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 2), (1, 3)),  # unigrams or bigrams
    # 'vect__tokenizer': (SkipgramTokenizer(3, 2), SkipgramTokenizer(2, 2), None),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__loss': ('squared_loss', 'hinge', 'log', 'epsilon_insensitive'),
    'clf__alpha': (0.0001, 0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__n_iter': (50, 80),
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

# Best score: 0.627
# Best parameters set:
# 	clf__alpha: 0.0001
# 	clf__loss: 'squared_loss'
# 	clf__n_iter: 50
# 	clf__penalty: 'elasticnet'
# 	vect__ngram_range: (1, 3)