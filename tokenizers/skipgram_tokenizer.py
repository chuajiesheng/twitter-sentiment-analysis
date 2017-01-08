from nltk.sentiment.util import mark_negation
from nltk.util import skipgrams

from .happy_tokenizer import Tokenizer


class SkipgramTokenizer(object):
    def __init__(self, n, k):
        self.tknzr = Tokenizer()
        self.n = n
        self.k = k

    def __call__(self, t):
        tokenised_tweet = list(self.tknzr.tokenize(t))
        negated_tweet = mark_negation(tokenised_tweet)
        list_of_skipgrams = list(skipgrams(negated_tweet, self.n, self.k))
        return list([' '.join(s) for s in list_of_skipgrams])

