import sys
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np


def log(message):
    log_file = 'instructional_sampling.log'
    with open(log_file, 'a') as f:
        f.write(message)
        f.flash()
        f.close()
    print message

# coding=utf-8
reload(sys)
sys.setdefaultencoding('utf-8')

# sc is an existing SparkContext.
sqlContext = SQLContext(sc)

failed_checks = 0

directory = "/Volumes/JS'S FIT/gnip-json"
datasets = sqlContext.read.json(directory)

file_count = datasets.where(datasets['verb'].isNull()).count()
# expecting 21888
assert file_count == 21888
if file_count != 21888:
    failed_checks += 1
    log('[error] file_count = {}'.format(file_count))

info_dataset = datasets.select('info')
info_dataset.registerTempTable('info')
all_tweets_count = info_dataset.select('info.activity_count').groupBy().sum('activity_count').collect()
# expecting 2682988

all_posts = datasets.where(datasets['verb'] == 'post')
all_posts_count = all_posts.count()
# expecting 1570398
assert all_posts_count == 1570398
if all_posts_count != 1570398:
    failed_checks += 1
    log('[error] all_posts_count = {}'.format(all_posts_count))
log('{} posts'.format(all_posts_count))

all_shares = datasets.where(datasets['verb'] == 'share')
all_shares_count = all_shares.count()
# expecting 1112590
assert all_shares_count == 1112590
if all_shares_count != 1112590:
    failed_checks += 1
    log('[error] all_shares_count = {}'.format(all_shares_count))
log('{} shares'.format(all_shares_count))

assert all_tweets_count[0][0] == all_posts_count + all_shares_count
if all_tweets_count[0][0] != all_posts_count + all_shares_count:
    failed_checks += 1
    log('[error] all_tweets_count = {}'.format(all_tweets_count[0][0]))

retweeted_post_ids = all_shares.select(all_shares['object.id'].alias('id')).rdd.map(lambda x: x.id).distinct()
post_ids = all_posts.select('id').rdd.map(lambda x: x.id).distinct()
keep_retweeted_post_ids = retweeted_post_ids.subtract(post_ids).collect()
retweeted_post_ids_count = retweeted_post_ids.count()

assert len(keep_retweeted_post_ids) < retweeted_post_ids_count
if len(keep_retweeted_post_ids) >= retweeted_post_ids_count:
    failed_checks += 1
    log('[error] keep_retweeted_post_ids = {}'.format(keep_retweeted_post_ids))
    log('[error] retweeted_post_ids_count = {}'.format(retweeted_post_ids_count))

exist_ = udf(lambda x: x in keep_retweeted_post_ids, BooleanType())
tweets_pool = all_posts.unionAll(all_shares.where(exist_(col('object.id')))).filter("twitter_lang = 'en'")

# check languages
languages = tweets_pool.select('twitter_lang').distinct()
languages_count = languages.count()
assert languages_count == 1
if languages_count != 1:
    failed_checks += 1
    log('[error] languages_count = {}'.format(languages_count))

language_retrieve = languages.first()
assert language_retrieve['twitter_lang'] == 'en'
if language_retrieve['twitter_lang'] != 'en':
    failed_checks += 1
    log('[error] language_retrieve = {}'.format(language_retrieve['twitter_lang']))

# validity check for tweets_pool
all_posts_ids = post_ids.collect()
validity_1 = udf(lambda x: x not in all_posts_ids, BooleanType())
validity_2 = udf(lambda x: x not in keep_retweeted_post_ids, BooleanType())
invalid_tweets_count = tweets_pool.where(tweets_pool['verb'] == 'post').where(validity_1(col('id'))).count()
assert invalid_tweets_count == 0
if invalid_tweets_count != 0:
    failed_checks += 1
    log('[error] invalid_tweets_count = {}'.format(invalid_tweets_count ))

invalid_tweets_count = tweets_pool.where(tweets_pool['verb'] == 'share').where(validity_2(col('object.id'))).count()
assert invalid_tweets_count == 0
if invalid_tweets_count != 0:
    failed_checks += 1
    log('[error] invalid_tweets_count = {}'.format(invalid_tweets_count ))

tweets_pool_count = tweets_pool.count()
tweets_pool_str_lengths = tweets_pool.select(length('body').alias('length')).rdd.map(lambda x: x.length).collect()
lengths_np = np.array(tweets_pool_str_lengths)
p = np.percentile(lengths_np, 20)
final_tweets_pool = tweets_pool.filter(length('body') >= p)
final_tweets_pool_count = final_tweets_pool.count()
percentage_kept = float(final_tweets_pool_count) / tweets_pool_count
assert percentage_kept > 0.8
if percentage_kept <= 0.8:
    failed_checks += 1
    log('[error] percentage_kept = {}'.format(percentage_kept))

sample_seed = 2016
number_of_instructional_samples = 30
sample_posts = final_tweets_pool.select(final_tweets_pool['id']).rdd.sortBy(lambda x: x.id).map(lambda x: x.id).takeSample(False, number_of_instructional_samples, sample_seed)
sample_posts_count = len(sample_posts)
log('{} sample posts'.format(sample_posts_count))

sample_posts_file = "./step_1/output/sample_posts.json"
sample_posts_jsons = final_tweets_pool[final_tweets_pool['id'].isin(sample_posts)].toJSON().collect()
with open(sample_posts_file, 'w') as f:
    for post in sample_posts_jsons:
        f.write(post)
        f.write('\n')

dev_seed = 20160717
number_of_dev_samples = 3000
dev_posts = final_tweets_pool.select(final_tweets_pool['id']).rdd.sortBy(lambda x: x.id).map(lambda x: x.id).takeSample(False, number_of_dev_samples, dev_seed)
dev_posts_count = len(dev_posts)
log('{} dev posts'.format(dev_posts_count))

dev_posts_file = "./step_1/output/dev_posts.json"
dev_posts_jsons = final_tweets_pool[final_tweets_pool['id'].isin(dev_posts)].toJSON().collect()
with open(dev_posts_file, 'w') as f:
    for post in dev_posts_jsons:
        f.write(post)
        f.write('\n')
