import numpy as np
import nltk
import sklearn
import tokenizers
import multiprocessing
import itertools
import functools
import pandas as pd
import scipy

INPUT_FILE = './step_4/input/sentiment.xlsx'
CV = 10
TRAIN_SIZE = 0.8
RANDOM_SEED = 42
K_BEST = 100
dataset = pd.read_excel(INPUT_FILE)

# re-sampling
negative_size = sum(dataset.sentiment == -1)
neutral_size = sum(dataset.sentiment == 0)
positive_size = sum(dataset.sentiment == 1)
print('Samples: \t{}\t{}\t{}'.format(negative_size, neutral_size, positive_size))

sample_size = min([negative_size, neutral_size, positive_size])

np.random.seed(RANDOM_SEED)

negative_dataset = dataset[dataset.sentiment == -1].index
neutral_dataset = dataset[dataset.sentiment == 0].index
positive_dataset = dataset[dataset.sentiment == 1].index

random_negative_indices = np.random.choice(negative_dataset, sample_size, replace=False)
random_neutral_indices = np.random.choice(neutral_dataset, sample_size, replace=False)
random_positive_indices = np.random.choice(positive_dataset, sample_size, replace=False)
indices = np.append(random_negative_indices, [random_neutral_indices, random_positive_indices])

x_text = dataset.loc[indices]['body']
x_liwc = dataset.loc[indices][['Analytic', 'Clout', 'Authentic', 'Tone', 'affect', 'posemo', 'negemo']]
y = dataset.loc[indices]['sentiment']

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
    x_text_train = x_text.iloc[train]
    x_liwc_train = x_liwc.iloc[train]
    y_train = y.iloc[train]

    x_text_test = x_text.iloc[test]
    x_liwc_test = x_liwc.iloc[test]
    y_test = y.iloc[test]

    vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer=TreebankTokenizer())
    x_text_train_vect = vect.fit_transform(x_text_train)

    tfidf = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False)
    x_text_train_tfidf = tfidf.fit_transform(x_text_train_vect)

    mutual_info = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.mutual_info_classif, k=K_BEST)
    x_text_train_k_best = mutual_info.fit_transform(x_text_train_tfidf, y_train)

    all_train_features = scipy.sparse.hstack((x_text_train_k_best, x_liwc_train)).A

    from sklearn.ensemble import *

    clf = RandomForestClassifier().fit(all_train_features, y_train)
    predicted = clf.predict(all_train_features)
    train_error = 1 - sklearn.metrics.accuracy_score(y_train, predicted)

    x_text_test_vect = vect.transform(x_text_test)
    x_text_test_tfidf = tfidf.transform(x_text_test_vect)
    x_text_test_k_best = mutual_info.transform(x_text_test_tfidf)
    all_test_features = scipy.sparse.hstack((x_text_test_k_best, x_liwc_test)).A
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
