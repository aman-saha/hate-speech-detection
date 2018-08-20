from stanfordcorenlp import StanfordCoreNLP
import pandas as pd
import json

nlp = StanfordCoreNLP(r'/Users/tommypawelski/Desktop/stanford-corenlp-full-2018-02-27')
data=pd.read_csv('labeled_data.csv',encoding = 'ISO-8859-1')

new_dict = dict()

for index, row in data.iterrows():
    tweet = str(row['tweet'])
    idx = str(row['index'])
    new_dict[idx]=nlp.dependency_parse(tweet)

json = json.dumps(new_dict)
f = open("dependency_dict.json","w")
f.write(json)

f.close()
nlp.close()
