import ast
from collections import OrderedDict


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
top_10_tweets = OrderedDict()

with open(RELEVANT_JSON) as f:
    for line in f:
        if len(line.strip()) < 1:
            continue

        t = ast.literal_eval(line)

        if t['object']['id'] == 'tag:search.twitter.com,2005:696116035502678016' \
                or t['object']['id'] == 'tag:search.twitter.com,2005:685869482112241664' \
                or t['object']['id'] == 'tag:search.twitter.com,2005:683100729498820608' \
                or t['object']['id'] == 'tag:search.twitter.com,2005:609058908594958336':
            continue

        top_10_tweets = add_to_sorted_dict(top_10_tweets, t['retweetCount'], t)

top_10_tweets = OrderedDict(sorted(top_10_tweets.items(), key=lambda t: t[0], reverse=True))
for k in top_10_tweets.keys():
    print(k, top_10_tweets[k]['id'])


