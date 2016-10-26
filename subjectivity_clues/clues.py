import os
import shlex


class Clues:
    DEFAULT_FILENAME = os.getcwd() + os.sep + 'subjectivity_clues' + os.sep + 'subjclueslen1-HLTEMNLP05.tff'

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

if __name__ == '__main__':
    c = Clues()
