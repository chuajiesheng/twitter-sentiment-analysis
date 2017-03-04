import os
from datetime import timezone, timedelta, datetime
import json
import csv

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


def get_sentiment_file(filename):
    relevance = dict()
    sentiment = dict()

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            relevance[row['id']] = row['relevance']
            sentiment[row['id']] = row['sentiment']

    return relevance, sentiment

# Read directory
DIRECTORY = './apology/input/tweets'
LABELLED_TWEETS = './apology/input/sentiment.csv'

relevance_labels, sentiment_labels = get_sentiment_file(LABELLED_TWEETS)
assert len(relevance_labels.keys()) == 2612018
assert len(sentiment_labels.keys()) == 2612018

tweets = set(list(relevance_labels.keys()) + list(sentiment_labels.keys()))
assert len(tweets) == 2612018

convert_to_full_path = lambda p: '{}/{}'.format(DIRECTORY, p)
files = list(map(convert_to_full_path, read_directory(DIRECTORY)))
assert len(files) == 21888

user_ids = [
    'id:twitter.com:16831827',
    'id:twitter.com:73752037',
    'id:twitter.com:2255835162',
    'id:twitter.com:330760001',
    'id:twitter.com:3171279230',
    'id:twitter.com:872992082',
    'id:twitter.com:2495416069',
    'id:twitter.com:878249012',
    'id:twitter.com:1354521667',
    'id:twitter.com:561837336',
    'id:twitter.com:608680669',
    'id:twitter.com:293209544',
    'id:twitter.com:537772227',
    'id:twitter.com:301766459',
    'id:twitter.com:1058847397',
    'id:twitter.com:2957633515',
    'id:twitter.com:363409296',
    'id:twitter.com:24332833',
    'id:twitter.com:1521040680',
    'id:twitter.com:785388690',
    'id:twitter.com:24847667',
    'id:twitter.com:1151362506',
    'id:twitter.com:174505543',
    'id:twitter.com:334359404',
    'id:twitter.com:532501558',
    'id:twitter.com:150933770',
    'id:twitter.com:40651949',
    'id:twitter.com:1287908988',
    'id:twitter.com:123437465',
    'id:twitter.com:559771943',
    'id:twitter.com:217613759',
    'id:twitter.com:44817044',
    'id:twitter.com:8145252',
    'id:twitter.com:265704335',
    'id:twitter.com:219668509',
    'id:twitter.com:4753383979',
    'id:twitter.com:4266289032',
    'id:twitter.com:4541449892',
    'id:twitter.com:4355670194',
    'id:twitter.com:3238061549',
    'id:twitter.com:19433195',
    'id:twitter.com:1314518538',
    'id:twitter.com:376615235',
    'id:twitter.com:30361852',
    'id:twitter.com:323525196',
    'id:twitter.com:61639235',
    'id:twitter.com:433034367',
    'id:twitter.com:1237970420',
    'id:twitter.com:223918713',
    'id:twitter.com:7377812',
    'id:twitter.com:2179292635',
    'id:twitter.com:3254110578',
    'id:twitter.com:1973486304',
    'id:twitter.com:746343968',
    'id:twitter.com:9312662',
    'id:twitter.com:21407872',
    'id:twitter.com:2778373431',
    'id:twitter.com:2329181064',
    'id:twitter.com:29791100',
    'id:twitter.com:146307561',
    'id:twitter.com:15170622',
    'id:twitter.com:372907535',

]
assert len(user_ids) == 62

tweets_user_mapping = dict()
for filename in files:
    tweets = read_file(filename)
    for t in tweets:
        if t['actor']['id'] in user_ids:
            body = t['body'].replace('\n', ' ').replace('\r', '').replace('"', '""')
            relevance = relevance_labels[t['id']] if t['id'] in relevance_labels.keys() else '?'
            sentiment = sentiment_labels[t['id']] if t['id'] in sentiment_labels.keys() else '?'
            print('"{}","{}",{},{},{},"{}",{},{}'.format(t['id'],
                                                    t['actor']['id'],
                                                    t['twitter_lang'],
                                                    t['verb'],
                                                    t['postedTime'],
                                                    body,
                                                    relevance,
                                                    sentiment))
