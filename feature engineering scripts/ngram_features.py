from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import string
import re
import pandas as pd
import numpy as np
from nltk import word_tokenize

data = pd.read_csv('cleaned_tweets.csv',encoding='utf-8')

stemmer = SnowballStemmer("english")
data['stemmed'] = data.clean_tweet.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))

#WORD LEVEL N-GRAMS (keep numbers as they represent unicode for emojis)
cv = CountVectorizer(stop_words='english', min_df=.002, max_df=.8, ngram_range=(2,2))
cv.fit(data.stemmed)
cv_mat = cv.transform(data.stemmed)

bigrams = pd.DataFrame(cv_mat.todense(), index=data['index'], columns=cv.get_feature_names())
bigrams = bigrams.add_prefix('word_bigrams:')
bigrams.to_csv('word_bigram_features.csv')


print ('Non-zero count:', cv_mat.nnz)
print ('Sparsity: %.2f%%' % (100.0 * cv_mat.nnz / (cv_mat.shape[0] * cv_mat.shape[1])))
oc = np.asarray(cv_mat.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'Term': cv.get_feature_names(), '# occurrences': oc})
counts_df.sort_values(by='# occurrences', ascending=False).head(20)


###########################################################################################################
#CHARACTER LEVEL N-GRAMS (remove unicode numbers first as they are not valuable at character level; also do not stem!)

data['char_stem'] = data.tweet.apply(lambda x: x.translate(str.maketrans('','',string.digits)))

cv_char = CountVectorizer(analyzer='char', stop_words='english',min_df=.002, max_df=.8,ngram_range=(2,2))
cv_char.fit(data.char_stem)
cv_char_mat = cv_char.transform(data.char_stem)


char_bigrams = pd.DataFrame(cv_char_mat.todense(), index=data['index'], columns=cv_char.get_feature_names())
char_bigrams = char_bigrams.add_prefix('char_bigrams:')

char_bigrams.to_csv('char_bigram_features.csv')

print ('Non-zero count:', cv_char_mat.nnz)
print ('Sparsity: %.2f%%' % (100.0 * cv_char_mat.nnz / (cv_char_mat.shape[0] * cv_char_mat.shape[1])))
oc2 = np.asarray(cv_char_mat.sum(axis=0)).ravel().tolist()
counts_df2 = pd.DataFrame({'Term': cv_char.get_feature_names(), '# occurrences': oc2})
counts_df2.sort_values(by='# occurrences', ascending=False).head(20)

###########################################################################################################
#TFIDF VALUES
cv = CountVectorizer(stop_words='english', min_df=.002, max_df=.8, ngram_range=(1,1))
cv.fit(data.stemmed)
cv_mat = cv.transform(data.stemmed)

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(cv_mat)

weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cv.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(80)
transformed_weights.toarray()

tf_idf =pd.DataFrame(transformed_weights.todense(), index=data['index'], columns=cv.get_feature_names())

tf_idf = tf_idf.add_prefix('tfidf:')

tf_idf.to_csv('tfidf_features.csv')
