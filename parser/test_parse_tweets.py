#!/usr/bin/env python3

import unittest
from parse_tweets import InvalidFormatError
from parse_tweets import TweetDatabase


class TestTweetDatabase(unittest.TestCase):

    def test_parse_line(self):
        test_string = '123\t456'
        tweet = TweetDatabase.parse_line(test_string)
        self.assertTrue(tweet.sid == '123')
        self.assertTrue(tweet.text == '456')

    def test_parse_line_without_second_part(self):
        test_string = '123'
        self.assertRaises(InvalidFormatError, TweetDatabase.parse_line, test_string)

if __name__ == '__main__':
    unittest.main()
