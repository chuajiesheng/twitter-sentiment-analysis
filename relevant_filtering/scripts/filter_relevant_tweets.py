import ast
from collections import OrderedDict
import csv
from utils import Reader
import os


def add_to_sorted_dict(ordered_dict, retweet_count, tweet):
    max_limit = 30
    tweet_id = tweet.data['object']['id']

    if tweet_id in ordered_dict.keys():
        if ordered_dict[tweet_id] < retweet_count:
            ordered_dict[tweet_id] = retweet_count
        return ordered_dict

    if len(ordered_dict) > 0:
        ordered_dict = OrderedDict(sorted(ordered_dict.items(), key=lambda t: t[1], reverse=True))

    current_min = min(ordered_dict.values(), default=0)
    if current_min >= retweet_count:
        return ordered_dict

    if len(ordered_dict) > (max_limit - 1):
        last_item = ordered_dict.popitem()

        min_count = min(ordered_dict.values())
        assert min_count >= last_item[1]

    assert(len(ordered_dict)) < max_limit

    ordered_dict[tweet_id] = retweet_count
    print('tweet', tweet_id, retweet_count)

    return ordered_dict

RELEVANT_TWEETS = './relevant_filtering/input/relevent_tweets.csv'
tweet_ids = []

with open(RELEVANT_TWEETS) as f:
    reader = csv.DictReader(f)
    for row in reader:
        tweet_ids.append(row['id'])

assert(len(tweet_ids) == 610319)

files = Reader.read_directory('./relevant_filtering/input/jsons')
top_tweets = OrderedDict()

for f in files:
    file_path = os.path.join('./relevant_filtering/input/jsons', f)
    tweets = Reader.read_file(file_path)
    for t in tweets:
        if t.is_post():
            continue

        if t.data['object']['id'] not in tweet_ids:
            continue

        top_tweets = add_to_sorted_dict(top_tweets, t.data['retweetCount'], t)

RELEVANT_JSON = './relevant_filtering/output/relevent_tweets_json.txt'

top_tweets = OrderedDict(sorted(top_tweets.items(), key=lambda t: t[1], reverse=True))
for k in top_tweets.keys():
    print(k, top_tweets[k])


