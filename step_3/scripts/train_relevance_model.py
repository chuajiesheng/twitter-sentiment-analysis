import numpy as np
import nltk
import sklearn
import tokenizers
import multiprocessing
import itertools
import functools
import pandas as pd
import scipy

INPUT_FILE = './step_3/input/relevance.xlsx'
CV = 10
TRAIN_SIZE = 0.8
RANDOM_SEED = 42
K_BEST = 100
dataset = pd.read_excel(INPUT_FILE)

x_text = dataset['body']
x_liwc = dataset[['Analytic', 'Clout', 'Authentic', 'Tone', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad']]
y = dataset['relevance']

total_train_error = 0.0
total_test_error = 0.0
runs = 0


class TreebankTokenizer(object):
    def __init__(self):
        self.treebank_word_tokenize = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize

    def __call__(self, doc):
        return self.treebank_word_tokenize(doc)


ss = sklearn.model_selection.ShuffleSplit(n_splits=CV, train_size=TRAIN_SIZE, test_size=None, random_state=RANDOM_SEED)
for train, test in ss.split(x_text):
    x_text_train = x_text.ix[train]
    x_liwc_train = x_liwc.ix[train]
    y_train = y.ix[train]

    x_text_test = x_text.ix[test]
    x_liwc_test = x_liwc.ix[test]
    y_test = y.ix[test]

    vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer=TreebankTokenizer())
    x_text_train_vect = vect.fit_transform(x_text_train)

    tfidf = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False)
    x_text_train_tfidf = tfidf.fit_transform(x_text_train_vect)

    mutual_info = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.mutual_info_classif, k=K_BEST)
    x_text_train_k_best = mutual_info.fit_transform(x_text_train_tfidf, y_train)

    all_train_features = scipy.sparse.hstack((x_text_train_k_best, x_liwc_train)).A

    clf = sklearn.linear_model.LogisticRegression().fit(all_train_features, y_train)
    predicted = clf.predict(all_train_features)
    train_error = 1 - sklearn.metrics.accuracy_score(y_train, predicted)

    x_text_test_vect = vect.transform(x_text_test)
    x_text_test_tfidf = tfidf.transform(x_text_test_vect)
    x_text_test_k_best = mutual_info.transform(x_text_test_tfidf)
    all_test_features = scipy.sparse.hstack((x_text_test_k_best, x_liwc_test)).A
    predicted = clf.predict(all_test_features)
    test_error = 1 - sklearn.metrics.accuracy_score(y_test, predicted)

    print('[{}] Accuracy: \t{}'.format(runs + 1, sklearn.metrics.accuracy_score(y_test, predicted)))
    print('[{}] Macro F1: \t{}'.format(runs + 1, sklearn.metrics.f1_score(y_test, predicted, average='macro')))

    total_train_error += train_error
    total_test_error += test_error
    runs += 1

print("[*] Average Accuracy: %0.3f" % 1 - (total_train_error / runs))
print("[*] Average Train Error: %0.3f" % (total_train_error / runs))
print("[*] Average Test Error: %0.3f" % (total_test_error / runs))
