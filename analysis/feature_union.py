import numpy as np
import nltk
import sklearn
import tokenizers
import multiprocessing
import itertools
import functools
import pandas
import scipy


class TreebankTokenizer(object):
    def __init__(self):
        self.treebank_word_tokenize = nltk.tokenize.treebank.TreebankWordTokenizer().tokenize

    def __call__(self, doc):
        return self.treebank_word_tokenize(doc)


class SentimentTokenizer(object):
    def __init__(self):
        self.sentiment_aware_tokenize = tokenizers.happy_tokenizer.Tokenizer().tokenize

    def __call__(self, doc):
        return self.sentiment_aware_tokenize(doc)


INPUT_FILE = './analysis/input/liwc_tweets.csv'
CV = 10
TRAIN_SIZE = 0.8
RANDOM_SEED = 42
K_BEST = 100
dataset = pandas.read_csv(INPUT_FILE)

x_text = dataset['body']
x_liwc = dataset[['Analytic', 'Clout', 'Authentic', 'Tone', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad']]
y = dataset['label']

total_train_error = 0.0
total_test_error = 0.0
runs = 0

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

    total_train_error += train_error
    total_test_error += test_error
    runs += 1

print(total_train_error / runs, total_test_error / runs)
exit(0)


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


def test_tokenizer(X, y, tokenizer, train_size, k_best):
    ss = sklearn.model_selection.ShuffleSplit(n_splits=10, train_size=(train_size / 100), test_size=None,
                                              random_state=42)
    total_train_error = 0.0
    total_test_error = 0.0
    total_f1 = 0.0
    runs = 0
    for train, test in ss.split(X, y):
        X_train = np.array(X)[train]
        y_train = y[train]

        X_test = np.array(X)[test]
        y_test = y[test]

        vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer=tokenizer)
        X_train_counts = vect.fit_transform(X_train)
        tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False).fit(X_train_counts)
        X_train_tfidf = tf_transformer.transform(X_train_counts)

        max_k = X_train_tfidf.shape[1]
        if k_best != 'all' and k_best > max_k:
            k_best = 'all'

        ch2 = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.mutual_info_classif, k=k_best)
        X_train_ch2 = ch2.fit_transform(X_train_tfidf, y_train)

        clf = sklearn.linear_model.LogisticRegression().fit(X_train_ch2, y_train)

        predicted = clf.predict(X_train_ch2)
        train_error = 1 - sklearn.metrics.accuracy_score(y_train, predicted)
        total_train_error += train_error

        X_test_counts = vect.transform(X_test)
        X_test_tfidf = tf_transformer.transform(X_test_counts)
        X_test_ch2 = ch2.transform(X_test_tfidf)
        predicted = clf.predict(X_test_ch2)

        test_error = 1 - sklearn.metrics.accuracy_score(y_test, predicted)
        total_test_error += test_error

        total_f1 += sklearn.metrics.f1_score(y_test, predicted, average='macro')
        runs += 1

    return total_train_error / runs, total_test_error / runs, total_f1 / runs




various_tokenizers = {
    'Treebank-style': TreebankTokenizer(),
    'Sentiment-aware': SentimentTokenizer()
}
train_sizes = list(range(60, 100, 10))
k_sizes = list(range(100, 10000, 200))
X, y = get_dataset()

TOKENIZER_F1_FILE = 'analysis/output/tokenizer_f1.csv'
TOKENIZER_ACC_FILE = 'analysis/output/tokenizer_accuracy.csv'

with open(TOKENIZER_F1_FILE, 'w') as f:
    f.writelines('tokenizer, train_size, k, f1\n')

with open(TOKENIZER_ACC_FILE, 'w') as f:
    f.writelines('tokenizer, train_size, k, train_error, test_error\n')


def train_and_output(X, y, tokenizer, train_size, k_best):
    tokenizer_name = tokenizer.__class__.__name__
    print('tokenizer={}, train_size={}, k_best={}'.format(tokenizer_name, train_size, k_best))
    average_train_error, average_test_error, average_f1 = test_tokenizer(X, y, tokenizer, train_size, k_best)
    with open('analysis/output/tokenizer_accuracy.csv', 'a') as acc_file:
        acc_file.write('{}, {}, {}, {:.3f}, {:.3f}\n'.format(tokenizer_name, train_size, k_best, average_train_error, average_test_error))
        acc_file.flush()
    with open('analysis/output/tokenizer_f1.csv', 'a') as f1_file:
        f1_file.write('{}, {}, {}, {:.3f}\n'.format(tokenizer_name, train_size, k_best, average_f1))
        f1_file.flush()

combi = itertools.product(various_tokenizers.values(), train_sizes, k_sizes)
with multiprocessing.Pool() as pool:
    p = pool.starmap(functools.partial(train_and_output, X, y), combi)
