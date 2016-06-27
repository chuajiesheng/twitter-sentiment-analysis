# coding=utf-8
import sys
import os
from utils import Reader

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')

    working_directory = os.getcwd()
    files = Reader.read_directory(working_directory)

    print 'verb,timestamp,body'

    for f in files:
        tweets = Reader.read_file(f)
        eng_tweets = filter(lambda t: t.language() == 'en', tweets)
        for tweet in tweets:
            body = tweet.body().replace('\n', ' ').replace('\r', '').replace('"', '""')
            print '{},{},"{}"'.format(tweet.verb(), tweet.timestamp(), body)

