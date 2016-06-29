# coding=utf-8

# OS-level import
import sys
import os
import code

# Data related import
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.tokenize import TweetTokenizer

# Project related object
from utils import Reader

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in tknzr.tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in tknzr.tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')

    working_directory = os.getcwd()
    files = Reader.read_directory(working_directory)
    print '{} files available'.format(len(files))

    # TODO: remove me
    files = files[:800]

    all_tweets = []
    for f in files:
        tweets = Reader.read_file(f)
        selected_tweets = filter(lambda t: t.is_post() and t.language() == 'en', tweets)
        texts = map(lambda t: t.body(), selected_tweets)
        all_tweets.extend(texts)

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in all_tweets:
        allwords_stemmed = tokenize_and_stem(i)  # for each item in 'all_tweets', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
    print 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame'

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.05, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(all_tweets)  # fit the vectorizer to synopses
    print 'td-idf matrix: {}'.format(tfidf_matrix.shape)

    terms = tfidf_vectorizer.get_feature_names()
    dist = 1 - cosine_similarity(tfidf_matrix)

    num_clusters = 10
    km = KMeans(n_clusters=num_clusters, verbose=0)
    # code.interact(local=dict(globals(), **locals()))
    km.fit(tfidf_matrix)

    clusters = km.labels_.tolist()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        print 'Cluster {} words: '.format(str(i)),

        for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
            print '{}'.format(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore')),

        print ''
