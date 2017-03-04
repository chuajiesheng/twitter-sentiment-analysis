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
print(files)
print(len(files))
assert len(files) == 21888

negative_tweet_ids = [
    'tag:search.twitter.com,2005:712442171333484544',
    'tag:search.twitter.com,2005:713096524083957760',
    'tag:search.twitter.com,2005:715319375390183429',
    'tag:search.twitter.com,2005:715648336447995906',
    'tag:search.twitter.com,2005:667851630721765376',
    'tag:search.twitter.com,2005:683359894695735297',
    'tag:search.twitter.com,2005:685265587660926976',
    'tag:search.twitter.com,2005:675039619591888896',
    'tag:search.twitter.com,2005:675054171545067520',
    'tag:search.twitter.com,2005:675461961803517952',
    'tag:search.twitter.com,2005:676612500654166016',
    'tag:search.twitter.com,2005:677183932006195200',
    'tag:search.twitter.com,2005:677341736079900672',
    'tag:search.twitter.com,2005:679487678283419648',
    'tag:search.twitter.com,2005:697186334901977090',
    'tag:search.twitter.com,2005:696738568002076672',
    'tag:search.twitter.com,2005:687684284841234432',
    'tag:search.twitter.com,2005:696882760896659456',
    'tag:search.twitter.com,2005:697500220016484352',
    'tag:search.twitter.com,2005:697956586107629572',
    'tag:search.twitter.com,2005:700857405865857024',
    'tag:search.twitter.com,2005:701278136193966080',
    'tag:search.twitter.com,2005:702334420552966144',
    'tag:search.twitter.com,2005:702715173061189632',
    'tag:search.twitter.com,2005:704827289956196352',
    'tag:search.twitter.com,2005:706263135200612352',
    'tag:search.twitter.com,2005:707336280586231808',
]
positive_tweet_ids = [
    'tag:search.twitter.com,2005:714214327612379136',
    'tag:search.twitter.com,2005:714992380336816129',
    'tag:search.twitter.com,2005:715539097251491840',
    'tag:search.twitter.com,2005:666170537366781952',
    'tag:search.twitter.com,2005:668669362916581376',
    'tag:search.twitter.com,2005:673008912593870848',
    'tag:search.twitter.com,2005:684472373379084288',
    'tag:search.twitter.com,2005:663098864426074112',
    'tag:search.twitter.com,2005:677207604465045504',
    'tag:search.twitter.com,2005:677778176672702465',
    'tag:search.twitter.com,2005:662489848113164288',
    'tag:search.twitter.com,2005:696760054536474624',
    'tag:search.twitter.com,2005:704434593479454720',
    'tag:search.twitter.com,2005:687677230533767168',
    'tag:search.twitter.com,2005:695452043910082560',
    'tag:search.twitter.com,2005:703229200153636864',
    'tag:search.twitter.com,2005:693986455656501248',
    'tag:search.twitter.com,2005:696755081803214848',
    'tag:search.twitter.com,2005:696755088031748096',
    'tag:search.twitter.com,2005:696755086073008128',
    'tag:search.twitter.com,2005:696755093526319104',
    'tag:search.twitter.com,2005:696755091634663425',
    'tag:search.twitter.com,2005:696830670195310592',
    'tag:search.twitter.com,2005:696848560768294912',
    'tag:search.twitter.com,2005:697158332277108736',
    'tag:search.twitter.com,2005:697479645780770816',
    'tag:search.twitter.com,2005:697525364638134272',
    'tag:search.twitter.com,2005:697550923250405376',
    'tag:search.twitter.com,2005:698962555918753793',
    'tag:search.twitter.com,2005:699600939053772800',
    'tag:search.twitter.com,2005:701158647930482688',
    'tag:search.twitter.com,2005:702584003472318464',
    'tag:search.twitter.com,2005:706213676873867265',
    'tag:search.twitter.com,2005:706560294307262468',
    'tag:search.twitter.com,2005:708027404342546432',
]
all_tweets = negative_tweet_ids + positive_tweet_ids

assert len(negative_tweet_ids) == 27
assert len(positive_tweet_ids) == 35

tweets_user_mapping = dict()
for filename in files:
    tweets = read_file(filename)
    for t in tweets:
        if t['id'] in all_tweets:
            print('{},{},{},{}'.format(t['id'], t['actor']['id'], t['actor']['link'], t['actor']['preferredUsername']))
