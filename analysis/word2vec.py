import numpy as np
from sklearn.model_selection import *
from sklearn.ensemble import *


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


# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


log.info('source load')
sources = {'./analysis/input/negative_tweets.txt': 'NEG', './analysis/input/neutral_tweets.txt': 'NEU', './analysis/input/positive_tweets.txt': 'POS'}

log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

log.info('D2V')
model = Doc2Vec(min_count=1, window=60, size=100, sample=1e-4, negative=5, workers=7)
model.build_vocab(sentences.to_array())

log.info('Epoch')
for epoch in range(10):
    log.info('EPOCH: {}'.format(epoch))
    model.train(sentences.sentences_perm())

import code; code.interact(local=dict(globals(), **locals()))

log.info('Model Save')
model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')

log.info('Sentiment')
X, Y = get_dataset()
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)
for train, test in ss.split(X, Y):
    size_train = len(train)
    size_test = len(test)

    train_arrays = numpy.zeros((size_train, 100))
    train_labels = numpy.zeros(size_train)

    X_train = np.array(X)[train]
    y_train = Y[train]

    X_test = np.array(X)[test]
    y_test = Y[test]

    for index, i in enumerate(train):
        if Y[i] == 1:
            prefix = 'POS_' + str(i - 1367 - 1367)
        elif Y[i] == 0:
            prefix = 'NEU_' + str(i - 1367)
        else:
            prefix = 'NEG_' + str(i)
        train_arrays[index] = model.docvecs[prefix]
        train_labels[index] = Y[i]

    test_arrays = numpy.zeros((size_test, 100))
    test_labels = numpy.zeros(size_test)

    for index, i in enumerate(test):
        if Y[i] == 1:
            prefix = 'POS_' + str(i - 1367 - 1367)
        elif Y[i] == 0:
            prefix = 'NEU_' + str(i - 1367)
        else:
            prefix = 'NEG_' + str(i)
        test_arrays[index] = model.docvecs[prefix]
        test_labels[index] = Y[i]

    log.info('Fitting')
    classifier = LogisticRegression(C=1.0, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=None, tol=0.00001)
    classifier.fit(train_arrays, train_labels)
    print(classifier.score(test_arrays, test_labels))

    clf = RandomForestClassifier(random_state=0, n_estimators=80, class_weight='auto').fit(train_arrays, train_labels)
    print(clf.score(test_arrays, test_labels))


def parts(str, current, elements):
    if len(str) < 1:
        return elements + [current]
    if current == '' or current.startswith(str[0]):
        return parts(str[1:], current + str[0], elements)
    return parts(str[1:], str[0], elements + [current])