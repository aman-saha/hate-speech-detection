#Author: Tommy Pawelski
#Created: July 13th 2018

import pandas as pd
import numpy as np
import nltk
import string
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from mlxtend.classifier import StackingCVClassifier

#read in each of the feature csv files
class_labels = pd.read_csv('labels.csv',encoding='utf-8')
weighted_tfidf_score = pd.read_csv('tfidf_scores.csv',encoding='utf-8')
sentiment_scores = pd.read_csv('sentiment_scores.csv',encoding='utf-8')
dependency_features = pd.read_csv('dependency_features.csv',encoding='utf-8')
char_bigrams = pd.read_csv('char_bigram_features.csv',encoding='utf-8')
word_bigrams = pd.read_csv('word_bigram_features.csv',encoding='utf-8')
tfidf_sparse_matrix = pd.read_csv('tfidf_features.csv',encoding='utf-8')

#merge all feature data sets based on 'index' column sentiment_scores, dependency_features, char_bigrams, word_bigrams
df_list=[class_labels, weighted_tfidf_score,sentiment_scores, dependency_features, char_bigrams, word_bigrams, tfidf_sparse_matrix]
master = df_list[0]
for df in df_list[1:]:
    master = master.merge(df, on='index')

master.columns.values
#ignore first two columns (index and tweet)

y=master.iloc[:,2] #class labels
X=master.iloc[:,3:] #all features


#create train and test sets: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##########################################################################################
#NOW WE CAN START MODELING
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn import model_selection
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

#Create a base training set to benchmark our performance (train set with hatespeech dictionary weighted tif-df score as only feature)
x_base = pd.DataFrame(X_train.loc[:,'weighted_TFIDF_scores'])
x_base_test = pd.DataFrame(X_test.loc[:,'weighted_TFIDF_scores'])

# created scaled version of training and test data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

#initialize models
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=.025)
xgb = XGBClassifier(learning_rate=.025, max_features=100)
mlp = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(80,40,40,10), activation='relu', random_state=1,learning_rate='adaptive', alpha=1e-6)
rf= RandomForestClassifier(n_estimators=100, max_features=500)
# 80,50,50,20

#asses model performances using 5-fold cross validation and f1-score micro aveage as metric
print("baseline model f1-score = ", cross_val_score(lr,x_base, y_train,cv=5,scoring="roc_auc").mean()) #benchmark model: linear regression using just tfidf score (weighted with hate dict)
print("gb cross validation f1-score = ", cross_val_score(gb,x_base, y_train,cv=5,scoring="f1_micro").mean()) #gradient boost with just tf-df score
print("rf cross validation f1-score = ", cross_val_score(rf,X_train,y_train,cv=5,scoring="f1_micro").mean()) #random forest with full train set (all features)
print("xgb cross validation f1-score = ", cross_val_score(xgb,X_train,y_train,cv=5,scoring="f1_micro").mean()) #xgboost with full train set (all features)
print("mlp cross validation f1-score = ", cross_val_score(mlp,X_train,y_train,cv=5,scoring="f1_micro").mean())

#initialize ensembles
estimators=[]
estimators.append(('mlp', mlp))
estimators.append(('rf', rf))
estimators.append(('xgb', xgb))

#voting ensemlbe
ensemble = VotingClassifier(estimators, voting='soft',weights=[1,1,1])
ensemble.fit(X_train, y_train)
pred = ensemble.predict(X_test)
print ('fscore:{0:.3f}'.format(f1_score(y_test, pred, average='micro')))

#meta classifier ensemble
stack = StackingCVClassifier(classifiers=[mlp, xgb, rf], cv=2,meta_classifier=lr, use_probas=True)
stack.fit(X_train.values, y_train.values)
pred2=stack.predict(X_test.values)
print ('fscore:{0:.3f}'.format(f1_score(y_test, pred2, average='micro')))

from sklearn.metrics import confusion_matrix
confusion_lr = confusion_matrix(y_test, pred)
print(confusion_lr)

####################################################################################################################
#REPORT AND PLOT MICRO-AVERAGE ROC AUC FOR EACH MODEL
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Compute micro-average ROC curve and ROC area
classifier = OneVsRestClassifier(lr)
classifier2= OneVsRestClassifier(gb)
classifier3 = OneVsRestClassifier(xgb)
y_score = classifier.fit(x_base, y_train).decision_function(x_base_test)
y_score2= classifier.fit(X_train, y_train).decision_function(X_test)
y_score3= classifier2.fit(X_train, y_train).decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr2 = dict()
tpr2 = dict()
roc_auc2 = dict()
fpr3 = dict()
tpr3 = dict()
roc_auc3 = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    fpr2[i], tpr2[i], _ = roc_curve(y_test[:, i], y_score2[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])
    fpr3[i], tpr3[i], _ = roc_curve(y_test[:, i], y_score3[:, i])
    roc_auc3[i] = auc(fpr3[i], tpr3[i])


fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
fpr2["micro"], tpr2["micro"], _ = roc_curve(y_test.ravel(), y_score2.ravel())
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])
fpr3["micro"], tpr3["micro"], _ = roc_curve(y_test.ravel(), y_score2.ravel())
roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
all_fpr2 = np.unique(np.concatenate([fpr2[i] for i in range(n_classes)]))
all_fpr3 = np.unique(np.concatenate([fpr3[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
mean_tpr2 = np.zeros_like(all_fpr2)
mean_tpr3 = np.zeros_like(all_fpr3)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr2 += interp(all_fpr2, fpr2[i], tpr2[i])
    mean_tpr3 += interp(all_fpr3, fpr3[i], tpr3[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
mean_tpr2 /= n_classes
mean_tpr3 /= n_classes

fpr["micro"] = all_fpr
tpr["micro"] = mean_tpr
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


fpr2["micro"] = all_fpr2
tpr2["micro"] = mean_tpr2
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])

fpr3["micro"] = all_fpr3
tpr3["micro"] = mean_tpr3
roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])

# Plot all ROC curves (cite: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='Base model micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='#EB7D3C', linestyle=':', linewidth=4)
plt.plot(fpr2["micro"], tpr2["micro"],
         label='LR micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc2["micro"]),
         color='#4674C1', linestyle=':', linewidth=4)
plt.plot(fpr3["micro"], tpr3["micro"],
         label='XGBoost micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc3["micro"]),
         color='#72AC48', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-Average ROC ')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.15),fancybox=True)
plt.show()

############## ROC values for just hate speech labels (class = 0)
plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='#EB7D3C',
         lw=lw, label='Base model ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot(fpr2[0], tpr2[0], color='#4674C1',
         lw=lw, label='LR ROC curve (area = %0.2f)' % roc_auc2[0])
plt.plot(fpr3[0], tpr3[0], color='#72AC48',
         lw=lw, label='XGBoost ROC curve (area = %0.2f)' % roc_auc3[0])
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for "Hatespeech" Label')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.15),fancybox=True)
plt.show()
