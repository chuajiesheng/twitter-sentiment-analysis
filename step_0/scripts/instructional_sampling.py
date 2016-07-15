# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# sc is an existing SparkContext.
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

directory = "/Volumes/JS'S FIT/gnip-json"
datasets = sqlContext.read.json(directory)

file_count = datasets.where(datasets['verb'].isNull()).count()
# expecting 21888
assert file_count == 21888

info_dataset = datasets.select('info')
info_dataset.registerTempTable('info')
all_tweets_count = info_dataset.select('info.activity_count').groupBy().sum('activity_count').collect()
# expecting 2682988

all_posts = datasets.where(datasets['verb'] == 'post')
all_posts_count = all_posts.count()
# expecting 1570398
assert all_posts_count == 1570398
print '{} posts'.format(all_posts_count)

all_shares = datasets.where(datasets['verb'] == 'share')
all_shares_count = all_shares.count()
# expecting 1112590
assert all_shares_count == 1112590
print '{} shares'.format(all_shares_count)

assert all_tweets_count[0][0] == all_posts_count + all_shares_count

retweeted_post_ids = all_shares.select(all_shares['object.id'].alias('id')).rdd.map(lambda x: x.id).distinct()
post_ids = all_posts.select('id').rdd.map(lambda x: x.id).distinct()
keep_retweeted_post_ids = retweeted_post_ids.subtract(post_ids).collect()
assert len(keep_retweeted_post_ids) < retweeted_post_ids.count()

from pyspark.sql.types import BooleanType
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
exist_ = udf(lambda x: x in keep_retweeted_post_ids, BooleanType())
tweets_pool = all_posts.unionAll(all_shares.where(exist_(col('object.id')))).filter("twitter_lang = 'en'")

# check languages
languages = tweets_pool.select('twitter_lang').distinct()
assert languages.count() == 1
assert languages.first()['twitter_lang'] == 'en'

# validity check for tweets_pool
all_posts_ids = post_ids.collect()
validity_1 = udf(lambda x: x not in all_posts_ids, BooleanType())
validity_2 = udf(lambda x: x not in keep_retweeted_post_ids, BooleanType())
invalid_tweets = tweets_pool.where(tweets_pool['verb'] == 'post').where(validity_1(col('id')))
assert invalid_tweets.count() == 0
invalid_tweets = tweets_pool.where(tweets_pool['verb'] == 'share').where(validity_2(col('object.id')))
assert invalid_tweets.count() == 0

from pyspark.sql.functions import length
tweets_pool_count = tweets_pool.count()
tweets_pool_str_lengths = tweets_pool.select(length('body').alias('length')).rdd.map(lambda x: x.length).collect()
import numpy as np
lengths_np = np.array(tweets_pool_str_lengths)
p = np.percentile(lengths_np, 20)
final_tweets_pool = tweets_pool.filter(length('body') >= p)
final_tweets_pool_count = final_tweets_pool.count()
assert (float(final_tweets_pool_count) / tweets_pool_count) > 0.8

sample_seed = 2016
number_of_instructional_samples = 30
sample_posts = final_tweets_pool.select(final_tweets_pool['id']).rdd.map(lambda x: x.id).takeSample(False, number_of_instructional_samples, sample_seed)
sample_posts_count = len(sample_posts)
print '{} sample posts'.format(sample_posts_count)

sample_posts_file = "./output/sample_posts.json"
sample_posts_jsons = sample_posts.toJSON().collect()
with open(sample_posts_file, 'a') as f:
	for post in sample_posts_jsons:
		f.write(post)
		f.write('\n')
