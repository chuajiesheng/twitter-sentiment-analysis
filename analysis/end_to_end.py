import numpy as np
import nltk
import sklearn
import tokenizers


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


tokenizers = {
    'Whitespace': WhitespaceTokenizer(),
    'Treebank-style': TreebankTokenizer(),
    'Sentiment-aware': SentimentTokenizer()
}
train_sizes = list(range(10, 100, 10))
X, y = get_dataset()

f1_csv = open('analysis/output/f1.csv', 'w')
acc_csv = open('analysis/output/accuracy.csv', 'w')

acc_csv.writelines('tokenizer, train_size, accuracy\n')
f1_csv.writelines('tokenizer, train_size, f1\n')

for keys in tokenizers.keys():
    tokenizer = tokenizers[keys]

    for train_size in train_sizes:
        ss = sklearn.model_selection.ShuffleSplit(n_splits=10, train_size=(train_size / 100), test_size=None, random_state=42)
        total_score = 0.0
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
            X_train_tf = tf_transformer.transform(X_train_counts)
            clf = sklearn.svm.LinearSVC().fit(X_train_tf, y_train)

            X_test_counts = vect.transform(X_test)
            X_test_tfidf = tf_transformer.transform(X_test_counts)
            predicted = clf.predict(X_test_tfidf)

            total_score += sklearn.metrics.accuracy_score(y_test, predicted)
            total_f1 += sklearn.metrics.f1_score(y_test, predicted, average='macro')
            runs += 1

        acc_csv.write('{}, {}%, {:.3f}\n'.format(keys, train_size, total_score / runs))
        f1_csv.write('{}, {}%, {:.3f}\n'.format(keys, train_size, total_f1 / runs))

f1_csv.close()
acc_csv.close()
