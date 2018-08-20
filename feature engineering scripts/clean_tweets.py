import pandas as pd
import re
import string

data=pd.read_csv('labeled_data.csv',encoding = 'ISO-8859-1')

clean_tweets = []
for index, row in data.iterrows():
    tweet = str(row['tweet']).lower()
    clean_tweets.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split()))

#print(clean_tweets)

data['clean_tweet'] = clean_tweets

data.to_csv("cleaned_tweets.csv", index=False)
