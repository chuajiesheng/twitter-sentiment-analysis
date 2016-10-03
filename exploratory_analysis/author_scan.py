import os
from utils import Reader
import code
import sys

author_dict = dict()


def extract_authors(tweets):
    # code.interact(local=dict(globals(), **locals()))

    for t in tweets:
        if t.is_post():
            actor = t.actor()
            create_key(actor['id'])
            increment_author(actor, t.is_post())

        elif t.is_share():
            original_tweet = t.data['object']
            actor = original_tweet['actor']
            create_key(actor['id'])
            increment_author(actor, t.is_post())
        else:
            print 'Neither post nor share:', t.id()


def increment_author(actor, is_post):
    dict_value = author_dict[actor['id']]

    dict_value[0] = actor['link']
    dict_value[1] = actor['preferredUsername']
    dict_value[2] = actor['displayName']

    if is_post:
        dict_value[3] += 1
    else:
        dict_value[4] += 1


def create_key(actor_id):
    if actor_id not in author_dict.keys():
        # link, username, display_name, post, post that gotten shared
        default_value = ['', '', '', 0, 0]
        author_dict[actor_id] = default_value


def print_all():
    for k in author_dict.keys():
        value = author_dict[k]
        print '"{}","{}","{}","{}",{},{}'.format(k, value[0], value[1], value[2], value[3], value[4])


if __name__ == '__main__':
    # coding=utf-8
    reload(sys)
    sys.setdefaultencoding('utf-8')

    working_directory = os.getcwd()
    files = Reader.read_directory(working_directory)

    for f in files:
        extract_authors(Reader.read_file(f))

    print_all()

    # code.interact(local=dict(globals(), **locals()))
