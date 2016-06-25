class Tweet:

    def __init__(self, json_object):
        self.data = json_object

    def timestamp(self):
        return self.data['postedTime']

    def verb(self):
        return self.data['verb']
