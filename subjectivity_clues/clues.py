import os
import shlex


class Clues:
    DEFAULT_FILENAME = os.getcwd() + os.sep + 'subjectivity_clues' + os.sep + 'subjclueslen1-HLTEMNLP05.tff'

    PRIORPOLARITY = {
        'positive': 1,
        'negative': -1,
        'both': 0,
        'neutral': 0
    }

    TYPE = {
        'strongsubj': 2,
        'weaksubj': 1
    }

    def __init__(self, filename=DEFAULT_FILENAME):
        lines = self.read_all(filename)
        self.lexicons = self.parse_clues(lines)

    @staticmethod
    def read_all(filename):
        with open(filename, 'r') as f:
            clues = f.readlines()
        return clues

    @staticmethod
    def parse_clues(lines):
        clues = dict()
        for l in lines:
            clue = dict(token.split('=') for token in shlex.split(l))
            word = clue['word1']
            clues[word] = clue
        return clues

    def calculate(self, sentence):
        related_words = 0
        total_subjectivity = 0
        total_priorpolarity = 0

        for w in sentence.split(' '):
            if w not in self.lexicons.keys():
                continue

            related_words += 1
            total_subjectivity += self.TYPE[self.lexicons[w]['type']]
            total_priorpolarity += self.PRIORPOLARITY[self.lexicons[w]['priorpolarity']]

        return {
            'sentence': sentence,
            'related_words': related_words,
            'total_subjectivity': total_subjectivity,
            'total_priorpolarity': total_priorpolarity
        }

    def calculate_related_words(self, sentence):
        related_words = 0

        for w in sentence.split(' '):
            if w not in self.lexicons.keys():
                continue

            related_words += 1

        return related_words

    def calculate_subjectivity(self, sentence):
        total_subjectivity = 0

        for w in sentence.split(' '):
            if w not in self.lexicons.keys():
                continue
            total_subjectivity += self.TYPE[self.lexicons[w]['type']]

        return total_subjectivity

    def calculate_priorpolarity(self, sentence):
        total_priorpolarity = 0

        for w in sentence.split(' '):
            if w not in self.lexicons.keys():
                continue

            total_priorpolarity += self.PRIORPOLARITY[self.lexicons[w]['priorpolarity']]

        return total_priorpolarity

    def calculate_relevant(self, sentence):
        total_subjectivity = 0
        total_priorpolarity = 0

        for w in sentence.split(' '):
            if w not in self.lexicons.keys():
                continue

            total_subjectivity += self.TYPE[self.lexicons[w]['type']]
            total_priorpolarity += self.PRIORPOLARITY[self.lexicons[w]['priorpolarity']]

        return total_subjectivity * total_priorpolarity

if __name__ == '__main__':
    c = Clues()
