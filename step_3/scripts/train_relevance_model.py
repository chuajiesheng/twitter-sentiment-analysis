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
SAMPLE_SIZE = 1500
dataset = pd.read_excel(INPUT_FILE)

# re-sampling
y_false = dataset[dataset.relevance == 0].index
np.random.seed(RANDOM_SEED)
random_y_false_indices = np.random.choice(y_false, SAMPLE_SIZE, replace=False)

indices = np.append(random_y_false_indices, np.array(dataset[dataset.relevance == 1].index))

x_text = dataset.loc[indices]['body'].reset_index(drop=True)
x_liwc = dataset.loc[indices][['Analytic', 'Clout', 'Authentic', 'Tone', 'affect', 'posemo', 'negemo']].reset_index(drop=True)
y = dataset.loc[indices]['relevance'].reset_index(drop=True)

total_accuracy = 0.0
total_train_error = 0.0
total_test_error = 0.0
total_f1 = 0.0
total_mcc = 0.0
runs = 0


class TreebankTokenizer(object):
    def __init__(self):
        self.treebank_word_tokenize = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize

    def __call__(self, doc):
        return self.treebank_word_tokenize(doc)


ss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=CV, train_size=TRAIN_SIZE, test_size=None, random_state=RANDOM_SEED)
for train, test in ss.split(x_text, y):
    x_text_train = x_text.loc[train]
    x_liwc_train = x_liwc.loc[train]
    y_train = y.loc[train]

    x_text_test = x_text.loc[test]
    x_liwc_test = x_liwc.loc[test]
    y_test = y.loc[test]

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
    print('[{}] MCC: \t{:.4f}'.format(runs + 1, sklearn.metrics.matthews_corrcoef(y_test, predicted)))
    print(sklearn.metrics.confusion_matrix(y_test, predicted))

    total_accuracy += sklearn.metrics.accuracy_score(y_test, predicted)
    total_train_error += train_error
    total_test_error += test_error
    total_f1 += sklearn.metrics.f1_score(y_test, predicted, average='macro')
    total_mcc += sklearn.metrics.matthews_corrcoef(y_test, predicted)
    runs += 1

print('[*] Average Train Accuracy/Error: \t{:.3f}\t{:.3f}'.format(1 - total_train_error / runs,
                                                                   total_train_error / runs))
print('[*] Average Test Accuracy/Error: \t{:.3f}\t{:.3f}'.format(total_accuracy / runs, total_test_error / runs))
print('[*] Average F1: \t\t\t{:.3f}'.format(total_f1 / runs))
print('[*] Average MCC: \t\t\t{:.3f}'.format(total_mcc / runs))
