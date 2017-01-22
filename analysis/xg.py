import numpy as np
from sklearn.metrics import *
from sklearn.feature_extraction.text import *
from sklearn.ensemble import *
from sklearn.model_selection import *
import xgboost as xgb
import matplotlib.pyplot as plt


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

    y = np.array([0] * 1367 + [1] * 1367 + [2] * 1367)
    return x, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


X, y = get_dataset()
X_train, X_test, y_train, y_test = split_dataset(X, y)
print('Train: \t\tX:{},\tY:{}'.format(len(X_train), y_train.shape[0]))
print('Test: \t\tX:{},\tY:{}'.format(len(X_test), y_test.shape[0]))

# train
count_vect = CountVectorizer(ngram_range=(1, 3))
X_train_counts = count_vect.fit_transform(X_train)

tf_transformer = TfidfTransformer(norm='l1', use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tf_transformer.transform(X_test_counts)

xg_train = xgb.DMatrix(X_train_tf, label=y_train)
xg_test = xgb.DMatrix(X_test_tfidf, label=y_test)

param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 1.3
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 3
watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 100

bst = xgb.train(param, xg_train, num_round)
yprob = bst.predict( xg_test ).reshape( y_test.shape[0], 3 )
ylabel = np.argmax(yprob, axis=1)
print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test)) ))

fig, ax = plt.subplots(1, 1)
xgb.plot_tree(bst, ax=ax)
fig.savefig('analysis/output/xg.png', dpi=600)


ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10)
total_score = 0.0
total_f1 = 0.0
runs = 0
for train, test in ss.split(X, y):
    X_train = np.array(X)[train]
    y_train = y[train]

    X_test = np.array(X)[test]
    y_test = y[test]

    count_vect = CountVectorizer(ngram_range=(1, 3))
    X_train_counts = count_vect.fit_transform(X_train)
    tf_transformer = TfidfTransformer(norm='l1', use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tf_transformer.transform(X_test_counts)

    xg_train = xgb.DMatrix(X_train_tf, label=y_train)
    xg_test = xgb.DMatrix(X_test_tfidf, label=y_test)

    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 1.3
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = 3
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 200

    bst = xgb.train(param, xg_train, num_round)
    yprob = bst.predict(xg_train).reshape(y_train.shape[0], 3)
    ylabel = np.argmax(yprob, axis=1)
    err = sum(int(ylabel[i]) != y_train[i] for i in range(len(y_train))) / float(len(y_train))

    yprob_test = bst.predict(xg_test).reshape(y_test.shape[0], 3)
    ylabel_test = np.argmax(yprob_test, axis=1)
    err_test = sum(int(ylabel_test[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test))

    fig, ax = plt.subplots(1, 1)
    xgb.plot_tree(bst, ax=ax)
    fig.savefig('analysis/output/{}.png'.format(runs), dpi=600)

    print('[{}] Accuracy: \t{:.3f}\t{:.3f}'.format(runs, 1 - err, 1 - err_test))

    total_score += (1 - err)
    runs += 1

print("[*] Accuracy: \t%0.3f" % (total_score / runs))
