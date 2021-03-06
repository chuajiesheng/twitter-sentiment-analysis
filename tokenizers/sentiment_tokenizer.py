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
        t = self.reduce_lengthening(t)
        tokens = t.split(' ')

        cleaned_tokens = []
        for token in tokens:
            token = self.replace_username(token)
            token = self.replace_link(token)
            cleaned_tokens.append(token)

        rebuild_str = ' '.join(cleaned_tokens)

        negated_tokens = mark_negation(list(self.tknzr.tokenize(rebuild_str)))
        list_of_trigrams = list([' '.join(s) for s in trigrams(negated_tokens)])
        return list_of_trigrams

