import numpy as np
from sklearn.metrics import *
from sklearn.feature_extraction.text import *
from sklearn.ensemble import *
from sklearn.pipeline import *
from sklearn.model_selection import *


class SkipgramTokenizer(object):
    SYMBOLS = [',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', ' ', '&', '$', '-', '|', '«', '`', ' ']

    def __init__(self, n, k):
        from nltk.tokenize.casual import TweetTokenizer
        from nltk.stem import WordNetLemmatizer

        self.tknzr = TweetTokenizer()
        self.wnl = WordNetLemmatizer()

        self.n = n
        self.k = k

    @staticmethod
    def remove_single_symbols(tokens):
        return [t.lower() for t in tokens if t.lower() not in SkipgramTokenizer.SYMBOLS]

    @staticmethod
    def remove_full_stops(tokens):
        return [t.lower() for t in tokens if t.lower() not in [' ', '.']]

    def __call__(self, t):
        from nltk.sentiment.util import mark_negation
        from nltk.util import skipgrams

        tokenised_tweet = self.remove_single_symbols(self.tknzr.tokenize(t))
        lemmatised_tweet = [self.wnl.lemmatize(w) for w in tokenised_tweet]
        three_to_six_chars_words = [w for w in lemmatised_tweet if (2 < len(w) < 7)]
        negated_tweet = mark_negation(three_to_six_chars_words)
        list_of_skipgrams = list(skipgrams(negated_tweet, self.n, self.k))
        features = list([' '.join(self.remove_full_stops(list(s))) for s in list_of_skipgrams])

        return features


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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tweets, target, test_size=0.2, random_state=12)

print('Train: \t\tX:{},\tY:{}'.format(len(X_train), y_train.shape[0]))
print('Test: \t\tX:{},\tY:{}'.format(len(X_test), y_test.shape[0]))

# train
count_vect = CountVectorizer(max_df=0.75, ngram_range=(1, 2), analyzer='word', stop_words='english')
X_train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer(norm='l1', use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
clf = RandomForestClassifier(random_state=0, n_estimators=80, class_weight='auto').fit(X_train_tf, y_train)

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tf_transformer.transform(X_test_counts)
predicted = clf.predict(X_test_tfidf)

print('Accuracy: \t{}'.format(accuracy_score(y_test, predicted)))
print('Macro F1: \t{}'.format(f1_score(y_test, predicted, average='macro')))
print(classification_report(y_test, predicted))

indices = np.argsort(clf.feature_importances_)[::-1][:2000] # 2000 features give 0.6116277334606841
X_train_new_tf = X_train_tf[:, indices]
clf = RandomForestClassifier(random_state=0, n_estimators=80, class_weight='auto').fit(X_train_new_tf, y_train)
predicted = clf.predict(X_test_tfidf[:, indices])

print('Accuracy: \t{}'.format(accuracy_score(y_test, predicted)))
print('Macro F1: \t{}'.format(f1_score(y_test, predicted, average='macro')))
print(classification_report(y_test, predicted))


ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)
total_score = 0.0
total_f1 = 0.0
runs = 0
for train, test in ss.split(tweets, target):
    X_train = np.array(tweets)[train]
    y_train = target[train]

    X_test = np.array(tweets)[test]
    y_test = target[test]

    X_train_counts = count_vect.transform(X_train)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train_new_tf = X_train_tf[:, indices]
    clf = RandomForestClassifier(random_state=0, n_estimators=80, class_weight='auto').fit(X_train_new_tf, y_train)

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tf_transformer.transform(X_test_counts)
    predicted = clf.predict(X_test_tfidf[:, indices])

    print('[{}] Accuracy: \t{}'.format(runs + 1, accuracy_score(y_test, predicted)))
    print('[{}] Macro F1: \t{}'.format(runs + 1, f1_score(y_test, predicted, average='macro')))

    total_score += accuracy_score(y_test, predicted)
    total_f1 += f1_score(y_test, predicted, average='macro')
    runs += 1

print("[*] Accuracy: %0.3f" % (total_score / runs))
print("[*] F1: %0.3f" % (total_f1 / runs))
