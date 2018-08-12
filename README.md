# Machine Learning and NLP methods for Automated Hate-Speech and Offensive Language Detection

## Overview ##


The dataset used for this project consists of Tweets labeled as `hate-speech`, `offensive language`, or `neither`. A more comprehensive description of the dataset is provided in `initial datasets directory`. The accompanying Python 3 scripts makes use of various Natural Language Processing and Machine Learning techniques to engineer various feature spaces (weighted TF-IDF scores, TF-IDF matrix, N-grams, sentiment scores, and dependency-based features) and trian various multi-class classification algorithms (Logistic Regression, XGBoost, and Multi-layer Perceprton) as well as Ensemble methods (weighted-average voting and stacking with LR meta-classifier). 


__NOTE:__ This is meant to be a customizable model, with the option to include any or all of the feature spaces engineered. To exclude a specific feature type, simply comment it out in lines 14-20 of [hate_speech_detection.py](https://github.com/tpawelski/hate-speech-detection/blob/master/hate_speech_detection.py).

## Instructions ##

The following describes how to run the hate-speech and offensive language detection model (described above) from scratch including all pre-processing and feature engineering steps:

- __STEP 1:__ Import the .csv file [labeled_data.csv](https://github.com/tpawelski/hate-speech-detection/blob/master/initial%20datasets/labeled_data.csv) (found in the `initial datasets` directory)

- __STEP 2:__  Open the `feature engineering scripts` directory and run [clean_tweets.py](https://github.com/tpawelski/hate-speech-detection/blob/master/feature%20engineering%20scripts/clean_tweets.py) which performs initial text corpus pre-processing. The output of this script is [cleaned_tweets.csv](https://github.com/tpawelski/hate-speech-detection/blob/master/initial%20datasets/cleaned_tweets.csv) which can also be found in the `initial datasets` directory.

- __STEP 3:__  Open and run each of the remaining scripts in the `feature engineering scripts` directory  which will create each of the feature subsets and output them as .csv files. The .csv file outputs from this step can be found in the `feature datasets` directory. The dictionaries required for this step can be found in the `dictionaries` directory. 

- __STEP 4:__ Open and run the script [hate_speech_detection.py](https://github.com/tpawelski/hate-speech-detection/blob/master/hate_speech_detection.py) which reads in the .csv files in the `feature datasets` directory, merges them into a single data frame, trains models to classify instances as either hate speech, offensive language, or neither, and performs model evaluation assesments on the testing set. 

- __STEP 5:__ Re-run steps 2-3 on any new raw text data, along with the best performing model trained in step 4, to detect instances of hate-speech and offensive language. 
