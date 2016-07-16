# Twitter Sentiment Analysis

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
