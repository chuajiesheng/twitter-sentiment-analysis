import os
from datetime import timezone, timedelta, datetime
import json
import csv

def read_directory(directory_name):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory_name):
        files.extend(filenames)
        break

    return files


def read_file(filename):
    is_tweet = lambda json_object: 'verb' in json_object and (json_object['verb'] == 'post' or json_object['verb'] == 'share')

    tweets = []

    with open(filename) as f:
        for line in f:
            if len(line.strip()) < 1:
                continue

            json_object = json.loads(line)
            if is_tweet(json_object):
                tweets.append(json_object)
            else:
                # this is a checksum line
                activity_count = int(json_object['info']['activity_count'])
                assert len(tweets) == activity_count

    return tweets


def get_sentiment_file(filename):
    relevance = dict()
    sentiment = dict()

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            relevance[row['id']] = row['relevance']
            sentiment[row['id']] = row['sentiment']

    return relevance, sentiment

# Read directory
DIRECTORY = './apology/input/tweets'
LABELLED_TWEETS = './apology/input/sentiment.csv'

relevance_labels, sentiment_labels = get_sentiment_file(LABELLED_TWEETS)
assert len(relevance_labels.keys()) == 2612018
assert len(sentiment_labels.keys()) == 2612018

tweets = set(list(relevance_labels.keys()) + list(sentiment_labels.keys()))
assert len(tweets) == 2612018

convert_to_full_path = lambda p: '{}/{}'.format(DIRECTORY, p)
files = list(map(convert_to_full_path, read_directory(DIRECTORY)))
assert len(files) == 21888

tweets_which_apology_user_retweet = [
    'tag:search.twitter.com,2005:712327576732225536',
    'tag:search.twitter.com,2005:714214203293179904',
    'tag:search.twitter.com,2005:666436770091966464',
    'tag:search.twitter.com,2005:666436793051586560',
    'tag:search.twitter.com,2005:666681909355921408',
    'tag:search.twitter.com,2005:661249353714114560',
    'tag:search.twitter.com,2005:661268150638551040',
    'tag:search.twitter.com,2005:661290475148718080',
    'tag:search.twitter.com,2005:668098362349109248',
    'tag:search.twitter.com,2005:661361943542964224',
    'tag:search.twitter.com,2005:668425385651085312',
    'tag:search.twitter.com,2005:668944592021000192',
    'tag:search.twitter.com,2005:669166203781455872',
    'tag:search.twitter.com,2005:669167952785182720',
    'tag:search.twitter.com,2005:669304410556747779',
    'tag:search.twitter.com,2005:672869317705338881',
    'tag:search.twitter.com,2005:672915562834604036',
    'tag:search.twitter.com,2005:673028381730967553',
    'tag:search.twitter.com,2005:683437670605832192',
    'tag:search.twitter.com,2005:683800073923248129',
    'tag:search.twitter.com,2005:684089604253835264',
    'tag:search.twitter.com,2005:684103445427568640',
    'tag:search.twitter.com,2005:684357724579172352',
    'tag:search.twitter.com,2005:684748134849851392',
    'tag:search.twitter.com,2005:684750221767622656',
    'tag:search.twitter.com,2005:684764678291992576',
    'tag:search.twitter.com,2005:684829669468823556',
    'tag:search.twitter.com,2005:685210256855535616',
    'tag:search.twitter.com,2005:685213530572877824',
    'tag:search.twitter.com,2005:663100092635873284',
    'tag:search.twitter.com,2005:685551710907969536',
    'tag:search.twitter.com,2005:673264152635301888',
    'tag:search.twitter.com,2005:673264101317984256',
    'tag:search.twitter.com,2005:673334129082163204',
    'tag:search.twitter.com,2005:673604256402972672',
    'tag:search.twitter.com,2005:673925703314882561',
    'tag:search.twitter.com,2005:674011278634033153',
    'tag:search.twitter.com,2005:674013337659817984',
    'tag:search.twitter.com,2005:674037657878548484',
    'tag:search.twitter.com,2005:661955089305763840',
    'tag:search.twitter.com,2005:674342033377632257',
    'tag:search.twitter.com,2005:674374346601832448',
    'tag:search.twitter.com,2005:674749698360627200',
    'tag:search.twitter.com,2005:675741612421591040',
    'tag:search.twitter.com,2005:674801917923491840',
    'tag:search.twitter.com,2005:662030538438524932',
    'tag:search.twitter.com,2005:675825265076011008',
    'tag:search.twitter.com,2005:675888290222985216',
    'tag:search.twitter.com,2005:675888313400733697',
    'tag:search.twitter.com,2005:674953612850798592',
    'tag:search.twitter.com,2005:674953546610135041',
    'tag:search.twitter.com,2005:675001499311874048',
    'tag:search.twitter.com,2005:675022070858784768',
    'tag:search.twitter.com,2005:675126448173285377',
    'tag:search.twitter.com,2005:676410019798228992',
    'tag:search.twitter.com,2005:676449585385050112',
    'tag:search.twitter.com,2005:676496351610462208',
    'tag:search.twitter.com,2005:676612282193805312',
    'tag:search.twitter.com,2005:676612500654166016',
    'tag:search.twitter.com,2005:676612740681629696',
    'tag:search.twitter.com,2005:676612687971815424',
    'tag:search.twitter.com,2005:677159699838537729',
    'tag:search.twitter.com,2005:677166726681894914',
    'tag:search.twitter.com,2005:677345003396538368',
    'tag:search.twitter.com,2005:677473107519668224',
    'tag:search.twitter.com,2005:677536454155902976',
    'tag:search.twitter.com,2005:677664343840964609',
    'tag:search.twitter.com,2005:677837310570926080',
    'tag:search.twitter.com,2005:677882717405483008',
    'tag:search.twitter.com,2005:662336823851687937',
    'tag:search.twitter.com,2005:678147090019696644',
    'tag:search.twitter.com,2005:679051704822812674',
    'tag:search.twitter.com,2005:679053728817786880',
    'tag:search.twitter.com,2005:679326881498402816',
    'tag:search.twitter.com,2005:679328675519688704',
    'tag:search.twitter.com,2005:679361154766864384',
    'tag:search.twitter.com,2005:679406803499790336',
    'tag:search.twitter.com,2005:679410059785412608',
    'tag:search.twitter.com,2005:679411164074364928',
    'tag:search.twitter.com,2005:679656758193844224',
    'tag:search.twitter.com,2005:679659990584524800',
    'tag:search.twitter.com,2005:679668097561804802',
    'tag:search.twitter.com,2005:679675710345187328',
    'tag:search.twitter.com,2005:679684927407915008',
    'tag:search.twitter.com,2005:679782872027414528',
    'tag:search.twitter.com,2005:660805737829031938',
    'tag:search.twitter.com,2005:680269508754313216',
    'tag:search.twitter.com,2005:681168071621505024',
    'tag:search.twitter.com,2005:682416457393790977',
    'tag:search.twitter.com,2005:682634713966710784',
    'tag:search.twitter.com,2005:682720429954478080',
    'tag:search.twitter.com,2005:682984746029219842',
    'tag:search.twitter.com,2005:683005335339057152',
    'tag:search.twitter.com,2005:683119584052928516',
    'tag:search.twitter.com,2005:662863694158778368',
    'tag:search.twitter.com,2005:662864373833187328',
    'tag:search.twitter.com,2005:686692093406908420',
    'tag:search.twitter.com,2005:686066364449185792',
    'tag:search.twitter.com,2005:689915708084322304',
    'tag:search.twitter.com,2005:693792799834128385',
    'tag:search.twitter.com,2005:695451890730061825',
    'tag:search.twitter.com,2005:697138282233270272',
    'tag:search.twitter.com,2005:698296941436915712',
    'tag:search.twitter.com,2005:707305816660041729',
    'tag:search.twitter.com,2005:685891379369623552',
    'tag:search.twitter.com,2005:688132381581000707',
    'tag:search.twitter.com,2005:689465493845970945',
    'tag:search.twitter.com,2005:690572489093685249',
    'tag:search.twitter.com,2005:698348606206533632',
    'tag:search.twitter.com,2005:699382262761185283',
    'tag:search.twitter.com,2005:661004255248179200',
    'tag:search.twitter.com,2005:703232032911073282',
    'tag:search.twitter.com,2005:686590264790532097',
    'tag:search.twitter.com,2005:686611264282312704',
    'tag:search.twitter.com,2005:663284249815064576',
    'tag:search.twitter.com,2005:663358715471491072',
    'tag:search.twitter.com,2005:688125487168655360',
    'tag:search.twitter.com,2005:689584117105582081',
    'tag:search.twitter.com,2005:690032571749634048',
    'tag:search.twitter.com,2005:690561518904020993',
    'tag:search.twitter.com,2005:690679667884355585',
    'tag:search.twitter.com,2005:691664060492967936',
    'tag:search.twitter.com,2005:692454124923850753',
    'tag:search.twitter.com,2005:694227259578830848',
    'tag:search.twitter.com,2005:694325538832388096',
    'tag:search.twitter.com,2005:694859934119411712',
    'tag:search.twitter.com,2005:694962511246786561',
    'tag:search.twitter.com,2005:695594218006380544',
    'tag:search.twitter.com,2005:696832277498691584',
    'tag:search.twitter.com,2005:697131088511180800',
    'tag:search.twitter.com,2005:697499410973003781',
    'tag:search.twitter.com,2005:698590221110083584',
    'tag:search.twitter.com,2005:698617120914087938',
    'tag:search.twitter.com,2005:698679396610273281',
    'tag:search.twitter.com,2005:699323109497839616',
    'tag:search.twitter.com,2005:699776797508960256',
    'tag:search.twitter.com,2005:700453430989488129',
    'tag:search.twitter.com,2005:702735528505053189',
    'tag:search.twitter.com,2005:703996139498901504',
    'tag:search.twitter.com,2005:707037259707469824',
    'tag:search.twitter.com,2005:707592169314365440',
    'tag:search.twitter.com,2005:665544298683764736',
    'tag:search.twitter.com,2005:710493389385539584',
    'tag:search.twitter.com,2005:710561725649702912',
    'tag:search.twitter.com,2005:710621553831112705',
]
assert len(tweets_which_apology_user_retweet) == 145

for filename in files:
    tweets = read_file(filename)
    for t in tweets:
        if t['id'] in tweets_which_apology_user_retweet:
            retweet_object = t['object']
            print('"{}","{}","{}","{}","{}"'.format(t['id'],
                                                    t['actor']['id'],
                                                    t['actor']['preferredUsername'],
                                                    retweet_object['actor']['id'],
                                                    retweet_object['actor']['preferredUsername']))
