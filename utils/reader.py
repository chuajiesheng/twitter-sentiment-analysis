import json
import os
from classes import Tweet


class Reader:
    def __init__(self):
        pass

    @staticmethod
    def read_directory(directory_name):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(directory_name):
            files.extend(filenames)
            break

        return files

    @staticmethod
    def read_file(filename):
        tweets = []

        with open(filename) as f:
            for line in f:
                if len(line.strip()) < 1:
                    continue

                json_object = json.loads(line)
                if Reader.is_tweet(json_object):
                    tweets.append(Tweet(json_object))
                else:
                    # this is a checksum line
                    activity_count = int(json_object['info']['activity_count'])
                    assert len(tweets) == activity_count

        return tweets

    @staticmethod
    def is_tweet(json_object):
        return 'verb' in json_object and (json_object['verb'] == 'post' or json_object['verb'] == 'share')
