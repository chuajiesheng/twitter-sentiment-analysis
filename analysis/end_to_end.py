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

        ch2 = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.mutual_info_classif, k=k_best)
        X_train_ch2 = ch2.fit_transform(X_train_tfidf, y_train)

        clf = sklearn.linear_model.LogisticRegression().fit(X_train_ch2, y_train)

        predicted = clf.predict(X_train_ch2)
        train_error = 1 - sklearn.metrics.accuracy_score(y_train, predicted)
        total_train_error += train_error

        X_test_counts = vect.transform(X_test)
        X_test_tfidf = tf_transformer.transform(X_test_counts)
        X_test_ch2 = ch2.fit_transform(X_test_tfidf, y_test)
        predicted = clf.predict(X_test_ch2)

        test_error = 1 - sklearn.metrics.accuracy_score(y_test, predicted)
        total_test_error += test_error

        total_f1 += sklearn.metrics.f1_score(y_test, predicted, average='macro')
        runs += 1

    average_train_error = total_train_error / runs
    average_test_error = total_test_error / runs
    average_f1 = total_f1 / runs

    return average_train_error, average_test_error, average_f1


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


various_tokenizers = {
    'Whitespace': WhitespaceTokenizer(),
    'Treebank-style': TreebankTokenizer(),
    'Sentiment-aware': SentimentTokenizer()
}
train_sizes = list(range(10, 100, 10))
k_sizes = [100]
X, y = get_dataset()

tokenizer_f1_csv = open('analysis/output/tokenizer_f1.csv', 'w')
tokenizer_acc_csv = open('analysis/output/tokenizer_accuracy.csv', 'w')

tokenizer_acc_csv.writelines('tokenizer, train_size, k, train_error, test_error\n')
tokenizer_f1_csv.writelines('tokenizer, train_size, k, f1\n')


for keys in various_tokenizers.keys():
    print('key={}'.format(keys))
    tok = various_tokenizers[keys]

    for size in train_sizes:
        print('train_size={}'.format(size))

        for k in k_sizes:
            print('k={}'.format(k))

            average_train_error, average_test_error, average_f1 = test_tokenizer(X, y, tok, size, k)
            tokenizer_acc_csv.write('{}, {}, {}, {:.3f}, {:.3f}\n'.format(keys, size, k, average_train_error, average_test_error))
            tokenizer_f1_csv.write('{}, {}, {}, {:.3f}\n'.format(keys, size, k, average_f1))

tokenizer_f1_csv.close()
tokenizer_acc_csv.close()

exit(0)

class SkipgramSentimentTokenizer(object):
    def __init__(self, n, k, negate=False):
        self.sentiment_aware_tokenize = tokenizers.happy_tokenizer.Tokenizer().tokenize
        self.n = n
        self.k = k
        self.negate = negate

    def __call__(self, doc):
        tokens = list(self.sentiment_aware_tokenize(doc))

        if self.negate:
            tokens = nltk.sentiment.util.mark_negation(tokens)

        if self.n == 1:
            return tokens

        skipgrams = list(nltk.skipgrams(tokens, self.n, self.k))
        return list([' '.join(s) for s in skipgrams])

features_extraction = {
    'Unigram': SkipgramSentimentTokenizer(1, 0),
    'Bigram': SkipgramSentimentTokenizer(2, 0),
    'Trigram': SkipgramSentimentTokenizer(3, 0),
    'Bigram with 1 skip': SkipgramSentimentTokenizer(2, 1),
    'Bigram with 2 skip': SkipgramSentimentTokenizer(2, 2),
    'Bigram with 3 skip': SkipgramSentimentTokenizer(2, 3),
    'Bigram with 4 skip': SkipgramSentimentTokenizer(2, 4),
    'Trigram with 1 skip': SkipgramSentimentTokenizer(3, 1),
    'Trigram with 2 skip': SkipgramSentimentTokenizer(3, 2),
    'Trigram with 3 skip': SkipgramSentimentTokenizer(3, 3),
    'Trigram with 4 skip': SkipgramSentimentTokenizer(3, 4),
    'Unigram (with negation)': SkipgramSentimentTokenizer(1, 0, negate=True),
    'Bigram (with negation)': SkipgramSentimentTokenizer(2, 0, negate=True),
    'Trigram (with negation)': SkipgramSentimentTokenizer(3, 0, negate=True),
    'Bigram with 1 skip (with negation)': SkipgramSentimentTokenizer(2, 1, negate=True),
    'Bigram with 2 skip (with negation)': SkipgramSentimentTokenizer(2, 2, negate=True),
    'Bigram with 3 skip (with negation)': SkipgramSentimentTokenizer(2, 3, negate=True),
    'Bigram with 4 skip (with negation)': SkipgramSentimentTokenizer(2, 4, negate=True),
    'Trigram with 1 skip (with negation)': SkipgramSentimentTokenizer(3, 1, negate=True),
    'Trigram with 2 skip (with negation)': SkipgramSentimentTokenizer(3, 2, negate=True),
    'Trigram with 3 skip (with negation)': SkipgramSentimentTokenizer(3, 3, negate=True),
    'Trigram with 4 skip (with negation)': SkipgramSentimentTokenizer(3, 4, negate=True),
}


features_acc_csv = open('analysis/output/features_accuracy.csv', 'w')
features_f1_csv = open('analysis/output/features_f1.csv', 'w')

features_acc_csv.writelines('features, train_size, train_error, test_error\n')
features_f1_csv.writelines('features, train_size, f1\n')

for keys in features_extraction.keys():
    print(keys)
    tok = features_extraction[keys]

    for size in train_sizes:
        average_train_error, average_test_error, average_f1 = test_tokenizer(X, y, tok, size)
        features_acc_csv.write('{}, {}%, {:.3f}, {:.3f}\n'.format(keys, size, average_train_error, average_test_error))
        features_f1_csv.write('{}, {}%, {:.3f}\n'.format(keys, size, average_f1))

features_acc_csv.close()
features_f1_csv.close()
