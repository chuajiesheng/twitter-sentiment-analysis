import sys
import json
from operator import *

from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *

import numpy as np


def expect(name, var, expected, op=eq):
    if op(var, expected):
        log('[checkpoint] {} = {}'.format(name, expected))
    else:
        log('[error] {} = {}'.format(name, expected))
        raise Exception(name)


def log(message):
    log_file = 'instructional_sampling.log'
    with open(log_file, 'a') as f:
        f.write(message)
        f.write('\n')
        f.flush()
        f.close()
    print message


def to_json(name, jsons):
    filename = '{}.json'.format(name)
    with open(filename, 'w') as f:
        for j in jsons:
            f.write(j)
            f.write('\n')


def to_csv(name, jsons):
    filename = '{}.csv'.format(name)
    with open(filename, 'w') as f:
        for tweet in jsons:
            t = json.loads(tweet)
            body = t['body'].replace('\n', ' ').replace('\r', '').replace('"', '""')
            f.write('"{}",{},{},"{}"\n'.format(t['id'], t['verb'], t['postedTime'], body))


# Make sure Python uses UTF-8 as tweets contains emoticon and unicode
reload(sys)
sys.setdefaultencoding('utf-8')

# Use SQLContext for better support
sqlContext = SQLContext(sc)

# Read GNIP's JSON file
directory = "tweets"
datasets = sqlContext.read.json(directory)
log('# Completed reading JSON files')

# Check checksum count
file_count = datasets.where(datasets['verb'].isNull()).count()
expect('file_count', file_count, 21888)

# Check post count
all_posts = datasets.where(datasets['verb'] == 'post')
all_posts_count = all_posts.count()
expect('all_posts_count', all_posts_count, 1570398)

# Check share count
all_shares = datasets.where(datasets['verb'] == 'share')
all_shares_count = all_shares.count()
expect('all_shares_count', all_shares_count, 1112590)

# Check dataset count
info_dataset = datasets.select('info')
info_dataset.registerTempTable('info')
all_tweets_count = info_dataset.select('info.activity_count').groupBy().sum('activity_count').collect()[0][0]
expect('all_tweets_count', all_tweets_count, 2682988)
expect('all_tweets_count', all_tweets_count, all_posts_count + all_shares_count)
log('# Completed validating tweets count')

# Remove post authored by @ChipotleTweet
# GET https://api.twitter.com/1.1/users/show.json?screen_name=ChipotleTweets to get id
chipotle_tweet = 'id:twitter.com:141341662'
all_posts_wo_chipotle = all_posts.filter("actor.id != '{}'".format(chipotle_tweet))
all_posts_w_chipotle_count = all_posts.filter("actor.id = '{}'".format(chipotle_tweet)).count()
expect('all_posts_wo_chipotle', all_posts_wo_chipotle.count(), all_posts_count - all_posts_w_chipotle_count)

# Remove share retweet of tweet by @ChipotleTweet
all_shares_wo_chipotle = all_shares.filter("object.actor.id != '{}'".format(chipotle_tweet))
all_shares_w_chipotle_count = all_shares.filter("object.actor.id = '{}'".format(chipotle_tweet)).count()
expect('all_shares_wo_chipotle', all_shares_wo_chipotle.count(), all_shares_count - all_shares_w_chipotle_count)

# Generate tweets pool with only English tweet
tweets_pool = all_posts_wo_chipotle.unionAll(all_shares_wo_chipotle).filter("twitter_lang = 'en'")
tweets_pool.cache()
tweets_pool_count = tweets_pool.count()
# Adding all post to all share will be greater than tweet pool because of non-English tweet
expected_tweets_pool_count = all_posts_count - all_posts_w_chipotle_count + \
                             all_shares_count - all_shares_w_chipotle_count
expect('tweets_pool_count', tweets_pool_count, expected_tweets_pool_count, op=lt)
log('# Completed constructing tweets pool')

# Check language of tweets
languages = tweets_pool.select('twitter_lang').distinct()
languages_count = languages.count()
language_check = languages.first()['twitter_lang']
expect('languages_count', languages_count, 1)
expect('language_check', language_check, 'en')
log('# Completed validating language variety')

# Take top 80% of tweets by length
tweets_pool_str_lengths = tweets_pool.select(length('body').alias('length')).rdd.map(lambda x: x.length).collect()
lengths_np = np.array(tweets_pool_str_lengths)
p = np.percentile(lengths_np, 20)

final_tweets_pool = tweets_pool.filter(length('body') >= p)
final_tweets_pool.cache()
tweets_pool.unpersist()

final_tweets_pool_count = final_tweets_pool.count()
percentage_kept = float(final_tweets_pool_count) / tweets_pool_count
expect('percentage_kept', percentage_kept, 0.8, op=gt)
log('# Completed sampling top 80% of tweets by body length')

# Sampling
final_tweets_ids = final_tweets_pool.select(final_tweets_pool['id']).rdd.sortBy(lambda x: x.id).map(lambda x: x.id)

# Sample tweets
sample_seed = 2016
number_of_instructional_samples = 30
sample_posts = final_tweets_ids.takeSample(False, number_of_instructional_samples, sample_seed)
sample_posts_count = len(sample_posts)
expect('sample_posts_count', sample_posts_count, number_of_instructional_samples)

sample_posts_file = "sample_posts"
sample_posts_jsons = final_tweets_pool[final_tweets_pool['id'].isin(sample_posts)].toJSON().collect()
log('Exporting sample post to {}'.format(sample_posts_file))
to_json(sample_posts_file, sample_posts_jsons)
to_csv(sample_posts_file, sample_posts_jsons)
log('# Completed exporting sample tweets')

# Development tweets
dev_seed = 20160717
number_of_dev_samples = 3000
dev_posts = final_tweets_ids.takeSample(False, number_of_dev_samples, dev_seed)
dev_posts_count = len(dev_posts)
expect('dev_posts_count', dev_posts_count, number_of_dev_samples)

dev_posts_file = "dev_posts"
dev_posts_jsons = final_tweets_pool[final_tweets_pool['id'].isin(dev_posts)].toJSON().collect()
to_json(dev_posts_file, dev_posts_jsons)
to_csv(dev_posts_file, dev_posts_jsons)
log('Exporting dev post to {}'.format(dev_posts_file))
log('# Completed exporting dev tweets')

# Inter-reliability test tweets
kappa_dev_seed = 0223
number_of_kappa_samples = 300
kappa_posts = sc.parallelize(dev_posts).takeSample(False, number_of_kappa_samples, kappa_dev_seed)
kappa_posts_count = len(kappa_posts)
expect('kappa_posts_count', kappa_posts_count, number_of_kappa_samples)
expect('kappa_posts_proper_subset', set(kappa_posts), set(dev_posts), op=set.issubset)

kappa_posts_file = "kappa_posts"
kappa_posts_jsons = final_tweets_pool[final_tweets_pool['id'].isin(kappa_posts)].toJSON().collect()
to_json(kappa_posts_file, kappa_posts_jsons)
to_csv(kappa_posts_file, kappa_posts_jsons)
log('Exporting kappa post to {}'.format(kappa_posts_file))
log('# Completed exporting kappa tweets')
