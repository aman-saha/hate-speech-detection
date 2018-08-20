import json
import pandas as pd
# Load the file
dependency_dict = json.loads(open("dependency_dict.json").read())
data=pd.read_csv('cleaned_tweets.csv',encoding = 'ISO-8859-1')

#find all dependency types found in our dataset, stored in set to ensure no repeats (all unique types)
dependency_types=set()
for key, values in dependency_dict.items():
    for v in list(values):
        dependency_types.add(list(v)[0])

#initialize columns (of all zeros) in dataframe for each of the dependency types
for type in dependency_types:
    data[str(type)] = 0

#print(data[0,])

for index, row in data.iterrows():
    tweet = str(row['tweet'])
    clean_tweet = str(row['clean_tweet'])
    idx = str(row['index'])
    dependeny_vec = dependency_dict[idx]
    #for each dependency type that tweet contains, add one to that dependecy column
    for dependency in dependeny_vec:
        data.loc[index, str(dependency[0])] += 1


data = data.add_prefix('dependecy:')
data.columns.values

data.to_csv("dependency_features.csv")
