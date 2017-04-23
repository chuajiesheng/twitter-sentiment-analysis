import csv
import os
import json
from collections import OrderedDict


def get_sentiment_file(filename):
    rows = []

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['relevance'] == '1':
                rows.append(row)

    return rows

# Read directory
LABELLED_TWEETS = './relevant_filtering/input/tweets.csv'
RELEVANT_TWEETS = './relevant_filtering/output/relevent_tweets.csv'

rows = get_sentiment_file(LABELLED_TWEETS)
print(len(rows))
assert len(rows) == 610319

if False:
    with open(RELEVANT_TWEETS, 'w') as csvfile:
        fieldnames = ['id', 'type', 'timestamp', 'body', 'Analytic', 'Clout', 'Authentic', 'Tone', 'affect', 'posemo',
                      'negemo', 'anx', 'anger', 'sad', 'relevance', 'sentiment', 'mention']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for r in rows:
            body = r['body']
            parts = body.split()
            for p in parts:
                if p.startswith('@'):
                    r['mention'] = '1'
                else:
                    r['mention'] = '0'

            writer.writerow(r)

tweet_ids = list(map(lambda row: row['id'], rows))
assert len(tweet_ids) == 610319


def read_directory(directory_name):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory_name):
        files.extend(filenames)
        break

    return files


def read_file(filename):
    is_tweet = lambda json_object: 'verb' in json_object and (json_object['verb'] == 'post' or json_object['verb'] == 'share')

    tweets = []

    with open(filename) as f:
        for line in f:
            if len(line.strip()) < 1:
                continue

            json_object = json.loads(line)
            if is_tweet(json_object):
                tweets.append(json_object)
            else:
                # this is a checksum line
                activity_count = int(json_object['info']['activity_count'])
                assert len(tweets) == activity_count

    return tweets


DIRECTORY = './relevant_filtering/input/tweets'
convert_to_full_path = lambda p: '{}/{}'.format(DIRECTORY, p)
files = list(map(convert_to_full_path, read_directory(DIRECTORY)))
assert len(files) == 21888


for filename in files:
    tweets = read_file(filename)
    for t in tweets:
        if t['id'] in tweet_ids:
            print(t)
