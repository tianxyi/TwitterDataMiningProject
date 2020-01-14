# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:18:19 2019

@author: tianx
"""

import main

NumberOfSample = 2000
        
emotionclass = TweetEmotionClassifier()
player = ["@KingJames", "@kobebryant", "@KDTrey5", "@StephenCurry30", "@SHAQ"]
#test = Tweets("@KingJames", 2000)
#test = Tweets("@kobebryant", 1500)
#test = Tweets("@KDTrey5", 20)
#test = Tweets("@StephenCurry30", 2000)
#test = Tweets("@SHAQ", 2000)
for people in player:
    test = Tweets(people, NumberOfSample)
    for item in test.get_tweets():
        item.attitude = emotionclass.classify(item.get_text())
    #tweet sentiment analysis
        print(item.get_text() + '\n ' + item.get_user_screen_name() + '\n '+item.get_date()  +'\n'+ item.get_attitude()+'\n ===================')  
    
    #whole player sentiment analysis
    print(test.pos_attitude_percentage(emotionclass))
    print(test.neg_attitude_percentage(emotionclass))
    
    #location distribution
    print(test.location_statistics())
    
    #race distribution
    print(test.race_statistics())
   
    #gender distribution
    print(test.gender_statistics())
