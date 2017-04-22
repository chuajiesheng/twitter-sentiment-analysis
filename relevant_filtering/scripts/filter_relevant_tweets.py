import csv

def get_sentiment_file(filename):
    rows = []

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['relevance'] == '1':
                rows.append(row)

    return rows

# Read directory
LABELLED_TWEETS = './relevant_filtering/input/tweets.csv'
RELEVANT_TWEETS = './relevant_filtering/output/relevent_tweets.csv'

rows = get_sentiment_file(LABELLED_TWEETS)
print(len(rows))
assert len(rows) == 610319

with open(RELEVANT_TWEETS, 'w') as csvfile:
    fieldnames = ['id', 'type', 'timestamp', 'body', 'Analytic', 'Clout', 'Authentic', 'Tone', 'affect', 'posemo',
                  'negemo', 'anx', 'anger', 'sad', 'relevance', 'sentiment', 'mention']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for r in rows:
        body = r['body']
        parts = body.split()
        for p in parts:
            if p.startswith('@'):
                r['mention'] = '1'
            else:
                r['mention'] = '0'

        writer.writerow(r)
