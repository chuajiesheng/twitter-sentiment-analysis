import ast
from collections import OrderedDict


def read_file(filename):
    tweets = []

    with open(filename) as f:
        for line in f:
            if len(line.strip()) < 1:
                continue

            json_object = ast.literal_eval(line)
            tweets.append(json_object)

    return tweets


def add_to_sorted_dict(ordered_dict, retweet_count, tweet):
    if len(ordered_dict) > 0:
        ordered_dict = OrderedDict(sorted(ordered_dict.items(), key=lambda t: t[0], reverse=True))

    current_min = min(ordered_dict, default=0)
    if current_min >= retweet_count:
        return ordered_dict

    if len(ordered_dict) > 14:
        last_item = ordered_dict.popitem()

        min_count = min(ordered_dict)
        assert min_count >= last_item[0]

    assert(len(ordered_dict)) < 15

    ordered_dict[retweet_count] = tweet

    return ordered_dict

RELEVANT_JSON = './relevant_filtering/output/relevent_tweets_json.txt'
tweets = read_file(RELEVANT_JSON)

tweet_parsed = 0
top_10_tweets = OrderedDict()

for t in tweets:
    tweet_parsed += 1
    top_10_tweets = add_to_sorted_dict(top_10_tweets, t['retweetCount'], t)

assert tweet_parsed == 610319

top_10_tweets = OrderedDict(sorted(top_10_tweets.items(), key=lambda t: t[0], reverse=True))
for k in top_10_tweets.keys():
    print(k, top_10_tweets[k]['id'])
