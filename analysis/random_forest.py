import numpy as np
from sklearn.metrics import *
from sklearn.feature_extraction.text import *
from sklearn.ensemble import *
from sklearn.model_selection import *


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


def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


X, y = get_dataset()
X_train, X_test, y_train, y_test = split_dataset(X, y)
print('Train: \t\tX:{},\tY:{}'.format(len(X_train), y_train.shape[0]))
print('Test: \t\tX:{},\tY:{}'.format(len(X_test), y_test.shape[0]))

# train
count_vect = CountVectorizer(ngram_range=(1, 2), analyzer='word', stop_words='english')
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

ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)
total_score = 0.0
total_f1 = 0.0
runs = 0
for train, test in ss.split(X, y):
    X_train = np.array(X)[train]
    y_train = y[train]

    X_test = np.array(X)[test]
    y_test = y[test]

    count_vect = CountVectorizer(ngram_range=(1, 2), analyzer='word', stop_words='english')
    X_train_counts = count_vect.fit_transform(X_train)
    tf_transformer = TfidfTransformer(norm='l1', use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    clf = RandomForestClassifier(random_state=0, n_estimators=80, class_weight='auto').fit(X_train_tf, y_train)

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tf_transformer.transform(X_test_counts)
    predicted = clf.predict(X_test_tfidf)

    print('[{}] Accuracy: \t{}'.format(runs + 1, accuracy_score(y_test, predicted)))
    print('[{}] Macro F1: \t{}'.format(runs + 1, f1_score(y_test, predicted, average='macro')))

    total_score += accuracy_score(y_test, predicted)
    total_f1 += f1_score(y_test, predicted, average='macro')
    runs += 1

print("[*] Accuracy: %0.3f" % (total_score / runs))
print("[*] F1: %0.3f" % (total_f1 / runs))
