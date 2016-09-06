# Notes

Small set of codes used during development and testing

## Using US-based S3

### Preparing AWS Spark cluster
```bash
aws s3 cp s3://chipotle-crisis/gnip-json.tar.gz .
hadoop fs -put gnip-json.tar.gz .
hdfs dfs -text gnip-json.tar.gz
hadoop fs -cat gnip-json.tar.gz | gzip -d | hadoop fs -put - tweets.json
```

### Moving files to tweets folder
```bash
# Copy tweets from S3 to HDFS
# Remember to use screen
hadoop fs -mkdir tweets
hdfs dfs -cp s3://chipotle-crisis/tweets_0.json tweets/tweet_0.json
hdfs dfs -cp s3://chipotle-crisis/tweets_1.json tweets/tweet_1.json
hdfs dfs -cp s3://chipotle-crisis/tweets_2.json tweets/tweet_2.json
hadoop fs -ls tweets
```

### Downloading and running the code
```bash
wget https://github.com/chuajiesheng/twitter-sentiment-analysis/archive/master.zip
unzip master.zip

cd twitter-sentiment-analysis
pyspark
execfile('step_1/scripts/instructional_sampling.py')
```

## Using SG-based S3

### Moving files to tweets folder
```bash
# Copy tweets from S3 to HDFS (SG version)
# Remember to use screen
hadoop fs -mkdir tweets
hdfs dfs -cp s3://chipotle-crisis-sg/tweets_0.json tweets/tweet_0.json
hdfs dfs -cp s3://chipotle-crisis-sg/tweets_1.json tweets/tweet_1.json
hdfs dfs -cp s3://chipotle-crisis-sg/tweets_2.json tweets/tweet_2.json
hadoop fs -ls tweets
```

### Downloading code
Remember to use screen
```bash
wget https://github.com/chuajiesheng/twitter-sentiment-analysis/archive/master.zip
```

### Reading dev tweets
```Python
directory = 'tweets.json'

# Loading sampled dev posts
hdfs dfs -cp s3://chipotle-crisis-sg/sampled/dev_posts.json dev_posts.json
sampled_json_file = 'dev_posts.json'
sampled_json = sqlContext.read.json(sampled_json_file)
sampled_json_id = sampled_json.select(sampled_json['id'])

sampled_set = set(sampled_json_id.collect())
dev_set = set(dev_posts)
len(sampled_set.symmetric_difference(dev_set))
```

### Reading kappa tweets
```bash
hdfs dfs -cp s3://chipotle-crisis-sg/sampled/kappa_posts.json kappa_posts.json
```

```Python
# Loading sampled kappa posts
sampled_kappa_json_file = 'kappa_posts.json'
sampled_kappa_json = stqlContext.read.json(sampled_kappa_json_file)
sampled_kappa_json_id = sampled_kappa_json.select(sampled_kappa_json['id'])

sampled_kappa_set = set(sampled_kappa_json_id.collect())
kappa_set = set(kappa_posts)
len(sampled_kappa_set.symmetric_difference(kappa_set))
```

### Copying gzipped tweets
```bash
# New gz file
hdfs dfs -cp s3://chipotle-crisis-sg/new/tweets.json.gz tweets.json.gz
```
