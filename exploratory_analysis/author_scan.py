import os
from utils import Reader
import code
import sys


def extract_authors(tweets):
    for t in tweets:
        if t.is_post():
            actor = t.actor()

            print '"{}","{}","{}","{}",{},{}'.format(actor['id'],
                                                     actor['link'],
                                                     actor['preferredUsername'],
                                                     actor['displayName'], 1, 0)

        elif t.is_share():
            original_tweet = t.data['object']
            actor = original_tweet['actor']

            print '"{}","{}","{}","{}",{},{}'.format(actor['id'],
                                                     actor['link'],
                                                     actor['preferredUsername'],
                                                     actor['displayName'], 0, 1)
        else:
            print 'Neither post nor share:', t.id()


if __name__ == '__main__':
    # coding=utf-8
    reload(sys)
    sys.setdefaultencoding('utf-8')

    working_directory = os.getcwd()
    files = Reader.read_directory(working_directory)

    for f in files:
        extract_authors(Reader.read_file(f))

    # code.interact(local=dict(globals(), **locals()))
