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
assert file_count == 21888
# expecting 21888

info_dataset = datasets.select('info')
info_dataset.registerTempTable('info')
all_tweets_count = info_dataset.select('info.activity_count').groupBy().sum('activity_count').collect()
# expecting 2682988

all_posts = datasets.where(datasets['verb'] == 'post')
all_posts_count = all_posts.count()
assert all_posts_count == 1570398
print '{} posts'.format(all_posts_count)
# expecting 1570398

all_shares = datasets.where(datasets['verb'] == 'share')
all_shares_count = all_shares.count()
assert all_shares_count == 1112590
print '{} shares'.format(all_shares_count)
# expecting 1112590

assert all_tweets_count[0][0] == all_posts_count + all_shares_count

sample_seed = 2016
fraction = 0.000022 # this give 30 samples
sample_posts = all_posts.sample(False, fraction, sample_seed)
sample_posts_count = sample_posts.count()
print '{} sample posts'.format(sample_posts_count)

sample_posts_file = "./output/sample_posts.json"
sample_posts_jsons = sample_posts.toJSON().collect()
with open(sample_posts_file, 'a') as f:
	for post in sample_posts_jsons:
		f.write(post)
		f.write('\n')
