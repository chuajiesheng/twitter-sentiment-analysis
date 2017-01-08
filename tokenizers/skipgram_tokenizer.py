from nltk.sentiment.util import mark_negation
from nltk.util import skipgrams
import re
from .happy_tokenizer import Tokenizer


class SkipgramTokenizer(object):
    def __init__(self, n, k):
        self.tknzr = Tokenizer()
        self.n = n
        self.k = k

    @staticmethod
    def reduce_lengthening(text):
        """
        Replace repeated character sequences of length 3 or greater with sequences
        of length 3.
        """
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1\1", text)

    def __call__(self, t):
        text = self.reduce_lengthening(t)
        tokens = list(self.tknzr.tokenize(text))
        negated_tokens = mark_negation(tokens)
        list_of_skipgrams = list(skipgrams(negated_tokens, self.n, self.k))
        return list([' '.join(s) for s in list_of_skipgrams])

