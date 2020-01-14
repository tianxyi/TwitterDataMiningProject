

import nltk
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
import json
from twitter import *
import gender_guesser.detector as gender
from datetime import *
from ethnicity import *
#import numpy as np
#import os
#import copy
#import difflib

class TweetEmotionClassifier:
    def __init__(self):
        """
        Intialize a new twitter emotion classifier self which is trained by dataset 
        in nltk.corpus. It is an sentiment analysis tool which inputs a text and return
        a positve or negative attitude.
        """
        tweets = [(map(lambda x:x.lower(), pos_tweet.split()), 'pos') for pos_tweet in twitter_samples.strings('positive_tweets.json')] + [(map(lambda x:x.lower(), neg_tweet.split()), 'neg') for neg_tweet in twitter_samples.strings('negative_tweets.json')]
        all_words = self._TweetEmotionClassifier__get_all_words()
        word_freq_dist = nltk.FreqDist(all_words)
        self.features = [word for (word, _) in word_freq_dist.most_common(600)]
        training_samples = [(self._TweetEmotionClassifier__extract_features(tweet), category) for (tweet, category) in tweets]
        self.classifier = nltk.NaiveBayesClassifier.train(training_samples)
        
    def __get_all_words(self):
        """
        helpfunction that read input from twitter classification with both positive tweet and
        negative tweet.
        """
        pos_words = [word.lower() for pos_tweet in twitter_samples.strings('positive_tweets.json') for word in pos_tweet.split()]
        neg_words = [word.lower() for neg_tweet in twitter_samples.strings('negative_tweets.json') for word in neg_tweet.split()]
        all_words = pos_words + neg_words
        return all_words
    
    def classify(self, tweet):
        """
        classify based on features of text
        """
        featurized_tweet = self._TweetEmotionClassifier__extract_features(map(lambda x:x.lower(), tweet.split()))
        return self.classifier.classify(featurized_tweet)
        
    def __extract_features(self, tweet):
        """
        extract feature of the text
        """
        tweet_words = set(tweet)
        extracted_features = {}
        for feature in self.features:
            extracted_features['contains(%s)' % feature] = (feature in tweet_words)
        return extracted_features

# Tweet data object
class Tweet:
    def __init__(self, tweet):
        """
        Intialize a new tweet class based on the filter return json file,
        and extract core attribute like text, date, location and etc to 
        benefit for the 
        """
       # self.emotion_classifier = emotion_classifier
        self.text = tweet['text']
        self.date = tweet['created_at']
        self.user_name = tweet['user']['name']
        self.user_screen_name = tweet['user']['screen_name']
        self.location = tweet['user']['location']
        self.attitude = "None"
    def get_text(self):
        return self.text
    
    def get_user_name(self):
        return self.user_name
    
    def get_user_screen_name(self):
        return self.user_screen_name
    
    def get_location(self):
        return self.location
    
    def get_date(self):
        return self.date
    
    def get_attitude(self):
        return self.attitude
    
class Tweets:
    def __init__(self, query, size=1000):
        """
        Initialize a new Tweets class that implement lots of core function
        of this project like using Twitter Search Api to fetch tweets,
        filtered tweets and all kinds of statistic analysis
        """
        credential = json.loads(open("credential.json", "r").read())
        auth = OAuth(credential['OAUTH_TOKEN'], credential['OAUTH_TOKEN_SECRET'], credential['CONSUMER_KEY'], credential['CONSUMER_SECRET'])
        twitter = Twitter(auth=auth)
        self.tweets = self._Tweets__fetch_tweets(query, twitter, size)
        self.ethnicity_predictor = Ethnicity().make_dicts()
       # self.emotion_classifier = emotion_classifier
        
    def __filter_non_informative_tweets(self, tweets):
        """
        filtering no informative tweets that has redundant symbol
        """
        filtered_tweets = []
        MINIMUM_WORD_COUNT = 6
        MAXIMUM_SHORT_SENTENCE_COUNT = 2
        for tweet in tweets:
            tweet_word_count = len(tweet.get_text().split())
            short_sentence_count = len([len(sentence) for sentence in tweet.get_text().split('\n') if len(sentence.split()) < 5 and len(sentence.split()) > 0])
            ata_count = len([char for char in tweet.get_text() if char == '@'])
            if tweet_word_count < MINIMUM_WORD_COUNT or short_sentence_count > MAXIMUM_SHORT_SENTENCE_COUNT or is_advertisement(tweet.get_text()):
                continue
            filtered_tweets.append(tweet)
        return filtered_tweets
    
    def __remove_duplicate_tweets(self, tweets):
        """
        filtering duplicate tweets to make analysis more accurate
        """
        filtered_tweets = []
        tweet_text_set = set()
        for tweet in tweets:
            if tweet.get_text() not in tweet_text_set:
                filtered_tweets.append(tweet)
                tweet_text_set.add(tweet.get_text())
        return filtered_tweets
    
    def __fetch_tweets(self, query, twitter, size):
        """
        using twitter search api to fetch tweets with english language. 
        And each payload is restricted to 100 based on the requirement.
        More over,doing the basic data preprocess about fetched information.
        """
        tweets = []
        try:
            max_id = twitter.search.tweets(q=query,lang='en',count=1)['search_metadata']['max_id']
        except:
            return tweets
        while (len(tweets)<size):
            load_size = 100
            if size - len(tweets) <= 100:
                load_size = size - len(tweets)
            try:
                tweet_search = twitter.search.tweets(q=query, lang='en',count=load_size, max_id=max_id)
            except:
                return tweets
            number_of_tweets = len(tweet_search['statuses'])
            if number_of_tweets == 0:
                break
            max_id = tweet_search['statuses'][number_of_tweets-1]['id'] - 1
            ###
            tweets += [Tweet(tweet) for tweet in tweet_search['statuses']]
            tweets = self._Tweets__filter_non_informative_tweets(tweets)
            tweets = self._Tweets__remove_duplicate_tweets(tweets)

            
        return tweets
        
    def get_tweets(self):
        return self.tweets
        
    def __predict_gender(self, names, screen_names):
        """
        predict gender distribution based on the Gender Detector
        """
        d = gender.Detector()
        predictions = []
        names_count = len(names)
        for i in range(names_count):
            prediction = d.get_gender(names[i])
            if prediction != 'female' and prediction != 'male':
                prediction = d.get_gender(screen_names[i])
            predictions.append(prediction)
    
        return predictions
        
    def gender_statistics(self):
        """
        the statistics result of gender distribution
        """
        names = [tweet.get_user_name() for tweet in self.tweets]
        screen_names = [tweet.get_user_screen_name() for tweet in self.tweets]
        
        predictions = self._Tweets__predict_gender(names, screen_names)
        stats = {}
        total = len(predictions)
        male_total, female_total, NA_total = 0, 0, 0
        for prediction in predictions:
            if prediction == 'male':
                male_total += 1
            elif prediction == 'female':
                female_total += 1
            else:
                NA_total += 1
        stats['male'] = male_total/total
        stats['female'] = female_total/total
        stats['NA'] = NA_total/total
        return stats
        
    def race_statistics(self):
        """
        predict race distribution based on the ethnicity_predictor
        """
        names = [tweet.get_user_name() for tweet in self.tweets]
        predictions = self.ethnicity_predictor.get(names)
        stats = {}
        total = len(names)
        for ethnicity in predictions["Ethnicity"]:
            if ethnicity not in stats:
                stats[ethnicity] = 1
            else:
                stats[ethnicity] += 1
        return {(k+"_percentage"): v/total for k, v in stats.items()}
    
    def __filter_invalid_location(self, locations):
        """
        filter invalid us location, based on the imported json file
        """
        cities_states_lookup_table = json.loads(open("cities_and_states.json", "r").read())
        filtered_locations = [location for location in locations if ', ' in location]
        filtered_locations = [location for location in filtered_locations if location.split(', ')[1] in cities_states_lookup_table]
        filtered_locations = [location for location in filtered_locations if location.split(', ')[0] in cities_states_lookup_table[location.split(', ')[1]]]
        return filtered_locations
    
    def location_statistics(self):
        """
        the statistics result of location distribution
        """
        locations = [tweet.get_location() for tweet in self.tweets if tweet.get_location() != '']
        non_filtered_locations_count = len(locations)
        locations = self._Tweets__filter_invalid_location(locations)
        stats = {}
        for location in locations:
            if location not in stats:
                stats[location] = 1
            else:
                stats[location] += 1
        if non_filtered_locations_count != 0:
            stats['unknown'] = non_filtered_locations_count - len(locations)
        return stats
    
    def __pos_attitude_percentage(self, attitudes):
        attitudes_total = len(attitudes)
        pos_total = 0
        for attitude in attitudes:
            if attitude == 'pos':
                pos_total += 1

        return pos_total/attitudes_total
    
    def pos_attitude_percentage(self, emotion_classifier):
        """
        using sentiment analysis tool to get the statistics result
        of positve attitude tweets
        """
        attitudes = []
        for tweet in self.get_tweets():
            attitudes.append(emotion_classifier.classify(tweet.get_text()))

        return self._Tweets__pos_attitude_percentage(attitudes)
    
    def __neg_attitude_percentage(self, attitudes):
        attitudes_total = len(attitudes)
        neg_total = 0
        for attitude in attitudes:
            if attitude == 'neg':
                neg_total += 1

        return neg_total/attitudes_total
    
    def neg_attitude_percentage(self, emotion_classifier):
        """
        using sentiment analysis tool to get the statistics result
        of negative attitude tweets
        """
        attitudes = []
        for tweet in self.get_tweets():
            attitudes.append(emotion_classifier.classify(tweet.get_text()))

        return self._Tweets__neg_attitude_percentage(attitudes)
    
    def overall_attitude(self, emotion_classifier):
        """
        based on statistic of pos/neg sentiment analysis to compute
        the overall attitude
        """
        attitudes = []
        for tweet in self.get_tweets():
            attitudes.append(emotion_classifier.classify(tweet.get_text()))
            
        pos_percentage = self._Tweets__pos_attitude_percentage(attitudes)
        neg_percentage = self._Tweets__neg_attitude_percentage(attitudes)
        overall_attitude = ''
        if pos_percentage > neg_percentage:
            return 'pos'
        if neg_percentage > pos_percentage:
            return 'neg'
        if neg_percentage == pos_percentage:
            return 'neutral'
        
def is_advertisement(tweet):
    """
    filtering advertisment tweet to make sentiment analysis more accurate.
    """
    is_advertisement = False
    advertisement_keywords = ['sale', 'sales', 'sell', 'buy', 'blackfriday', 'black friday', 'cyber monday', 'discount', 'discounts', 'ebay', 'amazon', 'best buy', 'bestbuy', 'walmart', 'target', 'used', 'new', 'condition']
    for advertisement_keyword in advertisement_keywords:
        if advertisement_keyword in tweet.lower():
            is_advertisement = True
            break
    return is_advertisement

#main function        
e9c = TweetEmotionClassifier()

#test = Tweets("@KingJames", 2000)
#test = Tweets("@kobebryant", 1500)
test = Tweets("@KDTrey5", 20)
#test = Tweets("@StephenCurry30", 2000)
#test = Tweets("@SHAQ", 2000)
for item in test.get_tweets():
    item.attitude = e9c.classify(item.get_text())
    print(item.get_text() + '\n ' + item.get_user_screen_name() + '\n '+item.get_date()  +'\n'+ item.get_attitude()+'\n ===================')  
print(test.pos_attitude_percentage(e9c))
print(test.neg_attitude_percentage(e9c))
print(test.location_statistics())
print(test.race_statistics())
print(test.gender_statistics())

#print(test.ethnicity_predictor)
