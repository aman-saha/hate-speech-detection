
import math
import nltk
import string
from collections import defaultdict
import math
import pandas as pd
import csv
import os
import pandas as pd
import numpy as np
import nltk
from sklearn import cross_validation

# hate database
dict1=pd.read_csv('hatebase_dict.csv', encoding = 'ISO-8859-1')
dict11 = dict1['dic']
dic1 = []
for row in dict11:
    row = row.strip("',")
    dic1.append(row)
    
#print(dic)
# negative words lexicon
dict2=pd.read_csv('negative word.csv', encoding = 'ISO-8859-1')
dict21 = dict2['dic']
dic2 = []
for row in dict21:
    row = row.strip("',")
    dic2.append(row)
    
# postive word lexicon
dict3=pd.read_csv('Postive words.csv', encoding = 'ISO-8859-1')
dict31 = dict3['dic']
dic3 = []
for row in dict31:
    row = row.strip("',")
    dic3.append(row)

hatedata = pd.read_csv('cleaned_tweets.csv')

tweet = hatedata['clean_tweet']
tweet1=tweet.str.split(" ")
hate = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    count = 0
    for j in tweet1[i]:
        for g in j.split(" "):
            if g in dic1:
                count+=1
        hate[i]=count

hatenor = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    l = len(tweet1[i])
    hatenor[i] = hate[i]/l

neg = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    ct = 0
    for j in tweet1[i]:
        for g in j.split(" "):
            if g in dic2:
                ct+=1
        neg[i]=ct

negnor = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    l = len(tweet1[i])
    negnor[i] = neg[i]/l

pos = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    ct1 = 0
    for j in tweet1[i]:
        for g in j.split(" "):
            if g in dic3:
                ct1+=1
        pos[i]=ct1

posnor = np.zeros(len(tweet))
for i in range(hatedata.shape[0]):
    l = len(tweet1[i])
    posnor[i] = pos[i]/l

hatedata["hate"] = hate
hatedata["hatenor"] = hatenor
hatedata["neg"] = neg
hatedata["negnor"] = negnor
hatedata["pos"] = pos
hatedata["posnor"] = posnor
hatedata.to_csv('sentiment_scores.csv')
