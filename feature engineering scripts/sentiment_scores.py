# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 20:13:29 2018

@author: kylin
"""

# read the data
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
dict1=pd.read_csv('/Users/kylin/Downloads/hatebase_dict.csv', encoding = 'ISO-8859-1')
dict11 = dict1['dic']
dic1 = []
for row in dict11:
    row = row.strip("',")
    dic1.append(row)
#print(dic)
# negative words
dict2=pd.read_csv('/Users/kylin/Downloads/negative word.csv', encoding = 'ISO-8859-1')
dict21 = dict2['dic']
dic2 = []
for row in dict21:
    row = row.strip("',")
    dic2.append(row)
# postive word
dict3=pd.read_csv('/Users/kylin/Downloads/Postive words.csv', encoding = 'ISO-8859-1')
dict31 = dict3['dic']
dic3 = []
for row in dict31:
    row = row.strip("',")
    dic3.append(row)

os.chdir('/Users/kylin/Downloads')
hatedata = pd.read_csv('/Users/kylin/Downloads/cleaned_tweets.csv')

tweet = hatedata['clean_tweet']
tweet1=tweet.str.split(" ")
a = np.zeros(len(tweet))
for i in range(24783):
    count = 0
    for j in tweet1[i]:
        for g in j.split(" "):
            if g in dic1:
                count+=1
        a[i]=count

#print(np.sum(a))

d = np.zeros(len(tweet))
for i in range(24783):
    l = len(tweet1[i])
    d[i] = a[i]/l

    
#print(np.sum(d))

b = np.zeros(len(tweet))
for i in range(24783):
    ct = 0
    for j in tweet1[i]:
        for g in j.split(" "):
            if g in dic2:
                ct+=1
        b[i]=ct
#print(np.sum(b))

e = np.zeros(len(tweet))
for i in range(24783):
    l = len(tweet1[i])
    e[i] = b[i]/l



c = np.zeros(len(tweet))
for i in range(24783):
    ct1 = 0
    for j in tweet1[i]:
        for g in j.split(" "):
            if g in dic3:
                ct1+=1
        c[i]=ct1
#print(np.sum(c))

f = np.zeros(len(tweet))
for i in range(24783):
    l = len(tweet1[i])
    f[i] = c[i]/l


hatedata["hate"] = a
hatedata["hatenor"] = d
hatedata["neg"] = b
hatedata["negnor"] = e
hatedata["pos"] = c
hatedata["posnor"] = f
hatedata.to_csv('/Users/kylin/Downloads/1.csv')