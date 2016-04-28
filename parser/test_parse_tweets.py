#!/usr/bin/env python3

import unittest
from parser import *


class TestTweetDatabase(unittest.TestCase):

    def test_parse_line(self):
        test_string = 'sid\tpositive\ttext'
        tweet = TweetDatabase.parse_line(test_string)
        self.assertEqual(tweet.sid, 'sid')
        self.assertEqual(tweet.sentiment, Sentiment.positive)
        self.assertEqual(tweet.text, 'text')

    def test_parse_line_without_first_part(self):
        test_string = '\t\ttext'
        self.assertRaises(InvalidTweetDataError, TweetDatabase.parse_line, test_string)

    def test_parse_line_without_second_part(self):
        test_string = '123'
        self.assertRaises(InvalidFormatError, TweetDatabase.parse_line, test_string)

    def test_read_db(self):
        test_db_path = 'test_fixtures/test_tweets.db'
        tweet_db = TweetDatabase(db_path=test_db_path)
        pairs = tweet_db.read_db()

        self.assertEqual(len(pairs), 3)

        tweet0 = pairs['638134980862828123']
        self.assertEqual(tweet0.sid, '638134980862828123')
        self.assertEqual(tweet0.sentiment, Sentiment.neutral)
        self.assertEqual(tweet0.text, '\'Happy People\' appeared on Saturday 29: http://t.co/123WgUhb #tgif')

        tweet1 = pairs['638156605448695123']
        self.assertEqual(tweet1.sid, '638156605448695123')
        self.assertEqual(tweet1.sentiment, Sentiment.positive)
        self.assertEqual(tweet1.text, 'Are you young enough to remember something attending the Grammys with people?')

        tweet2 = pairs['638162155250954123']
        self.assertEqual(tweet2.sid, '638162155250954123')
        self.assertEqual(tweet2.sentiment, Sentiment.negative)
        self.assertEqual(tweet2.text, '@123 do u enjoy his 2nd rate 123 bit? Honest ques. 2.0')

    def test_read_db_with_invalid_path(self):
        test_db_path = 'test_fixtures/some_wrong_path.db'
        tweet_db = TweetDatabase(db_path=test_db_path)
        self.assertRaises(FileNotFoundError, tweet_db.read_db)

if __name__ == '__main__':
    unittest.main()
