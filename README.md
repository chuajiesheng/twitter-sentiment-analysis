# Twitter Sentiment Analysis

## Analysis

### Step 0

1. link the GNIP json file to `step_0/input`
2. run `step_0/scripts/concat.py`

This will produce three (3) files

1. `tweets_0.json`
2. `tweets_1.json`
3. `tweets_2.json`

These file have been uploaded to S3, under the following folders

1. `s3://chipotle-crisis` (located in US)
2. `s3://chipotle-crisis-sg` (located in Singapore)

Currently, all the three files (frozen version) are also loaded to S3 in `s3://chipotle-crisis-final/step_0_results`

### Step 1

Initialise a Spark cluster in Amazon with the following configuration

1. emr-4.7.1
2. Spark 1.6.1
3. Hive 1.0.0
4. Hadoop 2.7.2
5. Hue 3.7.1
6. Zeppelin-Sandbox 0.5.6
7. Pig 0.14.0

The hardware configuration is as follows:

1. Master - m4.xlarge instance
2. Core - 4x m4.xlarge instance

After the cluster have started up, we need to copy the JSONs file to HDFS for Spark to consume. Do the following (US version):

```bash
# Remember to use screen
hadoop fs -mkdir tweets
hdfs dfs -cp s3://chipotle-crisis/tweets_0.json tweets/tweet_0.json
hdfs dfs -cp s3://chipotle-crisis/tweets_1.json tweets/tweet_1.json
hdfs dfs -cp s3://chipotle-crisis/tweets_2.json tweets/tweet_2.json
hadoop fs -ls tweets
```

For Spark cluster based in Singapore, use the following:

```bash
# Remember to use screen
hadoop fs -mkdir tweets
hdfs dfs -cp s3://chipotle-crisis-sg/tweets_0.json tweets/tweet_0.json
hdfs dfs -cp s3://chipotle-crisis-sg/tweets_1.json tweets/tweet_1.json
hdfs dfs -cp s3://chipotle-crisis-sg/tweets_2.json tweets/tweet_2.json
hadoop fs -ls tweets
```

Download the Python code from GitHub:

```bash
wget https://github.com/chuajiesheng/twitter-sentiment-analysis/archive/4483cecf8d9663a21bf3a1db7f2bb9f019ad4c4e.zip
unzip 4483cecf8d9663a21bf3a1db7f2bb9f019ad4c4e.zip
```

Then start up Spark and run the sampling script.
Note: I have realize that the `rdd.takeSample` return different sample on different machine.
```bash
# Note the hash
cd twitter-sentiment-analysis-4483cecf8d9663a21bf3a1db7f2bb9f019ad4c4e


# run Spark
pyspark 
execfile('step_1/scripts/instructional_sampling.py')
```

In the current working directory, which is `twitter-sentiment-analysis-4483cecf8d9663a21bf3a1db7f2bb9f019ad4c4e`.
You will find the following files:

Sample tweets for demo purpose
1. `sample_posts.csv`
2. `sample_posts.json`

Development tweets (3,000 tweets):
1. `dev_posts.csv`
2. `dev_posts.json`

Kappa tweets (300 tweets sampled from development tweets)
1. `kappa_posts.csv`
2. `kappa_posts.json`

Currently, all the six file (frozen version) have been loaded to S3 in `s3://chipotle-crisis-final/step_1_results`

## Using pySpark on Elastic MapReduce (EMR) in Amazon Web Services (AWS)

### Useful Commands

#### Copying of files from S3

```bash
aws s3 cp s3://<filepath> .
```

#### Placing files into HDFS from local

```bash
hadoop fs -put <filepath> .
```

#### Cat files from HDFS (zip file supported)

```bash
hdfs dfs -text <filepath>
```

#### Unzip file located in HDFS

```bash
hadoop fs -cat <filepath> | gzip -d | hadoop fs -put - <output>
```
