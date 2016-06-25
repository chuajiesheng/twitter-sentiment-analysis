import os
from utils import Reader

if __name__ == '__main__':
    working_directory = os.getcwd()
    files = Reader.read_directory(working_directory)
    for f in files:
        tweets = Reader.read_file(f)
        for tweet in tweets:
            print '{}, {}'.format(tweet.verb(), tweet.timestamp())

