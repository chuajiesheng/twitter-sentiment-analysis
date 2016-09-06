import sys
import json
from utils import Reader

directory = './step_0/input'
output_directory = './liwc_export/output'

def to_csv(name, jsons):
    filename = '{}/{}.csv'.format(output_directory, name)
    with open(filename, 'w') as f:
        for tweet in jsons:
            t = tweet.data
            body = t['body'].replace('\n', ' ').replace('\r', '').replace('"', '""')
            f.write('"{}",{},{},"{}"\n'.format(t['id'], t['verb'], t['postedTime'], body))


if __name__ == '__main__':
    print 'Note: This script is intended to run from twitter-sentiment-analysis root folder.'

    # Make sure Python uses UTF-8 as tweets contains emoticon and unicode
    reload(sys)
    sys.setdefaultencoding('utf-8')

    files = Reader.read_directory(directory)
    for f in files:
        tweets = Reader.read_file('{}/{}'.format(directory, f))
        selected_tweets = filter(lambda t: t.language() == 'en', tweets)
        filename = f[0:-5]
        to_csv(filename, selected_tweets)

        sys.stdout.write('.')
        sys.stdout.flush()
