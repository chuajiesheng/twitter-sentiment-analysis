import numpy as np
import nltk
import sklearn
import tokenizers
import multiprocessing
import itertools
import functools
import pandas as pd
import scipy
import os
import shlex

INPUT_FILE = './step_4/input/sentiment.xlsx'
CV = 10
TRAIN_SIZE = 0.8
RANDOM_SEED = 42
K_BEST = 100
SAMPLE_SIZE = 1500
dataset = pd.read_excel(INPUT_FILE)

# re-sampling
negative_size = sum(dataset.sentiment == -1)
neutral_size = sum(dataset.sentiment == 0)
positive_size = sum(dataset.sentiment == 1)

np.random.seed(RANDOM_SEED)

negative_dataset = dataset[dataset.sentiment == -1].index
neutral_dataset = dataset[dataset.sentiment == 0].index
positive_dataset = dataset[dataset.sentiment == 1].index

random_negative_indices = np.random.choice(negative_dataset, SAMPLE_SIZE, replace=False)
random_neutral_indices = np.random.choice(neutral_dataset, SAMPLE_SIZE, replace=False)
random_positive_indices = np.random.choice(positive_dataset, SAMPLE_SIZE, replace=False)
indices = np.append(random_negative_indices, [random_neutral_indices, random_positive_indices])
print('Total data size: {}'.format(len(indices)))


x_text = dataset.loc[indices]['body'].reset_index(drop=True)
x_liwc = dataset.loc[indices][['Analytic', 'Clout', 'Authentic', 'Tone', 'affect', 'posemo', 'negemo']].reset_index(drop=True)
y = dataset.loc[indices]['sentiment'].reset_index(drop=True)


def read_and_parse_clues():
    DEFAULT_FILENAME = os.getcwd() + os.sep + 'subjectivity_clues' + os.sep + 'subjclueslen1-HLTEMNLP05.tff'

    lines = None
    with open(DEFAULT_FILENAME, 'r') as f:
        lines = f.readlines()

    clues = dict()
    for l in lines:
        clue = dict(token.split('=') for token in shlex.split(l))
        word = clue['word1']
        clues[word] = clue

    return clues


def calculate_relevant(lexicons, sentence):
    PRIORPOLARITY = {
        'positive': 1,
        'negative': -1,
        'both': 0,
        'neutral': 0
    }

    TYPE = {
        'strongsubj': 2,
        'weaksubj': 1
    }

    total_score = 0

    for w in sentence.split(' '):
        if w not in lexicons.keys():
            continue

        total_score += PRIORPOLARITY[lexicons[w]['priorpolarity']] * TYPE[lexicons[w]['type']]

    return total_score


def read_and_parse_word_clusters():
    DEFAULT_FILENAME = os.getcwd() + os.sep + 'twitter_word_clusters' + os.sep + 'input' + os.sep + '50mpaths2.txt'

    with open(DEFAULT_FILENAME, 'r') as f:
        lines = f.readlines()

    word_clusters = dict()
    for l in lines:
        tokens = l.split('\t')
        cluster_path = tokens[0]
        word = tokens[1]

        word_clusters[word] = cluster_path

    return word_clusters


def tokenise(clusters, sentence):
    vector = dict()

    for w in sentence.split(' '):
        if w in clusters:
            path = clusters[w]
            if path in vector:
                vector[path] += 1
            else:
                vector[path] = 1

    return vector


lexicons = read_and_parse_clues()
x_subjectivity = x_text.apply(lambda row: calculate_relevant(lexicons, row)).rename('subjectivity')

clusters = read_and_parse_word_clusters()
dict_vectorizer = sklearn.feature_extraction.DictVectorizer()
x_clusters_dict = x_text.apply(lambda row: tokenise(clusters, row)).rename('word_clusters')
x_word_clusters = dict_vectorizer.fit_transform(x_clusters_dict)

x_features = pd.concat([x_liwc, x_subjectivity, pd.DataFrame(x_word_clusters.todense())], axis=1)

total_accuracy = 0.0
total_train_error = 0.0
total_test_error = 0.0
total_f1 = 0.0
runs = 0


class TreebankTokenizer(object):
    def __init__(self):
        self.treebank_word_tokenize = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize

    def __call__(self, doc):
        return self.treebank_word_tokenize(doc)


ss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=CV, train_size=TRAIN_SIZE, test_size=None, random_state=RANDOM_SEED)
for train, test in ss.split(x_text, y):
    x_text_train = x_text.loc[train]
    x_features_train = x_features.loc[train]
    y_train = y.loc[train]

    x_text_test = x_text.loc[test]
    x_features_test = x_features.loc[test]
    y_test = y.loc[test]

    vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer=TreebankTokenizer())
    x_text_train_vect = vect.fit_transform(x_text_train)

    tfidf = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False)
    x_text_train_tfidf = tfidf.fit_transform(x_text_train_vect)

    mutual_info = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.mutual_info_classif, k=K_BEST)
    x_text_train_k_best = mutual_info.fit_transform(x_text_train_tfidf, y_train)

    all_train_features = scipy.sparse.hstack((x_text_train_k_best, x_features_train)).A

    from sklearn.ensemble import *

    clf = RandomForestClassifier(n_estimators=500).fit(all_train_features, y_train)
    predicted = clf.predict(all_train_features)
    train_error = 1 - sklearn.metrics.accuracy_score(y_train, predicted)

    x_text_test_vect = vect.transform(x_text_test)
    x_text_test_tfidf = tfidf.transform(x_text_test_vect)
    x_text_test_k_best = mutual_info.transform(x_text_test_tfidf)
    all_test_features = scipy.sparse.hstack((x_text_test_k_best, x_features_test)).A
    predicted = clf.predict(all_test_features)
    test_error = 1 - sklearn.metrics.accuracy_score(y_test, predicted)

    print('[{}] Accuracy: \t{:.4f}'.format(runs + 1, sklearn.metrics.accuracy_score(y_test, predicted)))
    print('[{}] Macro F1: \t{:.4f}'.format(runs + 1, sklearn.metrics.f1_score(y_test, predicted, average='macro')))
    print(sklearn.metrics.confusion_matrix(y_test, predicted))

    total_accuracy += sklearn.metrics.accuracy_score(y_test, predicted)
    total_train_error += train_error
    total_test_error += test_error
    total_f1 += sklearn.metrics.f1_score(y_test, predicted, average='macro')
    runs += 1

print('[*] Average Train Accuracy/Error: \t{:.3f}\t{:.3f}'.format(1 - total_train_error / runs,
                                                                   total_train_error / runs))
print('[*] Average Test Accuracy/Error: \t{:.3f}\t{:.3f}'.format(total_accuracy / runs, total_test_error / runs))
print('[*] Average F1: \t\t\t{:.3f}'.format(total_f1 / runs))
