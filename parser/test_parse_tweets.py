#!/usr/bin/env python3

import unittest
from parse_tweets import InvalidFormatError
from parse_tweets import TweetDatabase


class TestTweetDatabase(unittest.TestCase):

    def test_parse_line(self):
        test_string = '123\t456'
        tweet = TweetDatabase.parse_line(test_string)
        self.assertEqual(tweet.sid, '123')
        self.assertEqual(tweet.text, '456')

    def test_parse_line_without_first_part(self):
        test_string = '\t123'
        tweet = TweetDatabase.parse_line(test_string)
        self.assertEqual(tweet.sid, '')
        self.assertEqual(tweet.text, '123')

    def test_parse_line_without_second_part(self):
        test_string = '123'
        self.assertRaises(InvalidFormatError, TweetDatabase.parse_line, test_string)

    def test_read_db(self):
        test_db_path = 'test_fixtures/test_tweets.db'
        tweet_db = TweetDatabase(db_path=test_db_path)
        pairs = tweet_db.read_db()

        self.assertEqual(len(pairs), 3)

        tweet0 = pairs['637641175948712345']
        self.assertEqual(tweet0.sid, '637641175948712345')
        self.assertEqual(tweet0.text, 'Not Available')

        tweet1 = pairs['637651487762551234']
        self.assertEqual(tweet1.sid, '637651487762551234')
        self.assertEqual(tweet1.text, '@ah some tweet')

        tweet2 = pairs['637666734300901234']
        self.assertEqual(tweet2.sid, '637666734300901234')
        self.assertEqual(tweet2.text, 'Not Available')

    def test_read_db_with_invalid_path(self):
        test_db_path = 'test_fixtures/some_wrong_path.db'
        tweet_db = TweetDatabase(db_path=test_db_path)
        self.assertRaises(FileNotFoundError, tweet_db.read_db)


if __name__ == '__main__':
    unittest.main()
