from nltk.sentiment.util import mark_negation
from nltk.util import trigrams
import re
import validators
from .happy_tokenizer import Tokenizer


class SentimentTokenizer(object):
    def __init__(self):
        self.tknzr = Tokenizer()

    @staticmethod
    def reduce_lengthening(text):
        """
        Replace repeated character sequences of length 3 or greater with sequences
        of length 3.
        """
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1\1", text)

    @staticmethod
    def replace_username(token):
        return '@__user__' if token.startswith('@') else token

    @staticmethod
    def replace_link(token):
        return '__url__' if validators.url(token) else token

    def __call__(self, t):
        text = self.reduce_lengthening(t)
        tokens = list(self.tknzr.tokenize(text))

        cleaned_tokens = []
        for t in tokens:
            t = self.replace_username(t)
            t = self.replace_link(t)
            cleaned_tokens.append(t)

        negated_tokens = mark_negation(cleaned_tokens)
        list_of_skipgrams = list(trigrams(negated_tokens))
        return list([' '.join(s) for s in list_of_skipgrams])

