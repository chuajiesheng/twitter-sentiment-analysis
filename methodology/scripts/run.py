import numpy as np
import nltk
import sklearn
import pandas as pd
import os
import shlex
import hashlib

INPUT_FILE = './methodology/input/complete_set.xlsx'
OUTPUT_FILE = './methodology/output/sentiment.csv'
CV = 10
TRAIN_SIZE = 0.8
RANDOM_SEED = 42
SAMPLE_SIZE = 1500
CLASSIFY = True
dataset = pd.read_excel(INPUT_FILE)

np.random.seed(RANDOM_SEED)

random_indices = lambda dataset: np.random.choice(dataset.index, SAMPLE_SIZE, replace=False)
indices = np.append(random_indices(dataset[dataset.sentiment == -1]),
                    [random_indices(dataset[dataset.sentiment == 0]),
                     random_indices(dataset[dataset.sentiment == 1])])
print('Total data size: {}'.format(len(indices)))

X = dataset.loc[indices][['body', 'Analytic', 'Clout', 'Authentic', 'Tone', 'affect', 'posemo', 'negemo']].reset_index(
    drop=True)
y = dataset.loc[indices]['sentiment'].reset_index(drop=True)


class WordExtractor(sklearn.base.TransformerMixin):
    def transform(self, X, **transform_params):
        return X['body']

    def fit(self, X, y=None, **fit_params):
        return self


class LiwcFeatureExtractor(sklearn.base.TransformerMixin):
    def transform(self, X, **transform_params):
        return X[['Analytic', 'Clout', 'Authentic', 'Tone', 'affect', 'posemo', 'negemo']]

    def fit(self, X, y=None, **fit_params):
        return self


class WhitespaceTokenizer(object):
    def __init__(self):
        pass

    def __call__(self, doc):
        return doc.split(' ')


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


class SubjectivityTransformer(sklearn.base.TransformerMixin):
    @classmethod
    def read_and_parse_clues(cls):
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

    @classmethod
    def calculate_relevant(cls, lexicons, sentence):
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

    def __init__(self):
        self._lexicons = self.read_and_parse_clues()

    def transform(self, X, **transform_params):
        return pd.DataFrame(X.apply(lambda row: self.calculate_relevant(self._lexicons, row)))

    def fit(self, X, y=None, **fit_params):
        return self


class WordClusterTransformer(sklearn.base.TransformerMixin):
    @classmethod
    def read_and_parse_word_clusters(cls):
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

    @classmethod
    def tokenise(cls, clusters, sentence):
        vector = dict()

        for w in sentence.split(' '):
            if w in clusters:
                path = clusters[w]
                if path in vector:
                    vector[path] += 1
                else:
                    vector[path] = 1

        return vector

    def __init__(self):
        self._word_clusters = self.read_and_parse_word_clusters()

    def transform(self, X, **transform_params):
        return X.apply(lambda row: self.tokenise(self._word_clusters, row))

    def fit(self, X, y=None, **fit_params):
        return self


def sha(filename):
    BUF_SIZE = 65536

    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('features', FeatureUnion([
        # ('liwc', Pipeline([
        #     ('extract', LiwcFeatureExtractor())
        # ])),
        # ('subjectivity', Pipeline([
        #     ('extract', WordExtractor()),
        #     ('subjectivity_vec', SubjectivityTransformer())
        # ])),
        # ('word_cluster', Pipeline([
        #     ('extract', WordExtractor()),
        #     ('word_vec', WordClusterTransformer()),
        #     ('dict_vec', sklearn.feature_extraction.DictVectorizer())
        # ])),
        ('words', Pipeline([
            ('extract', WordExtractor()),
            ('count_vec', sklearn.feature_extraction.text.CountVectorizer(tokenizer=TreebankTokenizer())),
            ('td_idf', sklearn.feature_extraction.text.TfidfTransformer(use_idf=False))
        ]))
    ])),
    ('classifier', RandomForestClassifier(n_estimators=50))
])

total_accuracy = 0.0
total_train_error = 0.0
total_test_error = 0.0
total_f1 = 0.0
runs = 0

print('train_size,train_accuracy,train_error,train_f1,test_accuracy,test_error,test_f1')
for size in range(10, 100, 10):
    train_size = size / 100
    ss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=CV, train_size=train_size, test_size=None,
                                                        random_state=RANDOM_SEED)

    total_train_accuracy = 0.0
    total_train_error = 0.0
    total_train_f1 = 0.0

    total_test_accuracy = 0.0
    total_test_error = 0.0
    total_test_f1 = 0.0

    runs = 0

    for train, test in ss.split(X, y):
        pipeline.fit(X.loc[train], y.loc[train])
        predicted_train = pipeline.predict(X.loc[train])
        predicted_test = pipeline.predict(X.loc[test])

        train_accuracy_score = sklearn.metrics.accuracy_score(y.loc[train], predicted_train)
        train_error = (1 - train_accuracy_score)
        train_f1_score = sklearn.metrics.f1_score(y.loc[train], predicted_train, average='macro')

        total_train_accuracy += train_accuracy_score
        total_train_error += train_error
        total_train_f1 += train_f1_score

        test_accuracy_score = sklearn.metrics.accuracy_score(y.loc[test], predicted_test)
        test_error = 1 - test_accuracy_score
        test_f1_score = sklearn.metrics.f1_score(y.loc[test], predicted_test, average='macro')

        total_test_accuracy += test_accuracy_score
        total_test_error += test_error
        total_test_f1 += test_f1_score

        runs += 1

    print('{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
        train_size,
        total_train_accuracy / runs,
        total_train_error / runs,
        total_train_f1 / runs,
        total_test_accuracy / runs,
        total_test_error / runs,
        total_test_f1 / runs))
