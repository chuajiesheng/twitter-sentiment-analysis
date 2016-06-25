class Tweet:

    def __init__(self, json_object):
        self.data = json_object

    def timestamp(self):
        return self.data['postedTime']

    def verb(self):
        return self.data['verb']

    def is_post(self):
        return self.verb() == 'post'

    def is_share(self):
        return self.verb() == 'share'

    def generator(self):
        """return the application which created this tweets (e.g. 'Twitter for iPhone', 'Twitter Web Client')"""
        return self.data['generator']

    def retweet_count(self):
        return self.data['retweetCount']

    def favorites_count(self):
        return self.data['favoritesCount']

    def actor(self):
        """return twitter person object (which contains the location and etc)"""
        return self.data['actor']

    def location(self):
        """if there exist, return a twitter place object indicating where the tweet is created"""
        if 'location' not in self.data.keys():
            return None

        return self.data['location']

    def body(self):
        """return the display text of the tweet (if is a retweet, this contains 'RT @someone ...')"""
        return self.data['body']

    def language(self):
        return self.data['twitter_lang']

    def reply_to(self):
        """return the link (of the tweet) in which this tweet is replying to"""
        if 'inReplyTo' not in self.data.keys():
            return None

        return self.data['inReplyTo']
