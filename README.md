# Machine Learning and NLP methods for Automated Hate-Speech Detection

The following .md file describes  how to run the Hate-Speech Detection Model:

STEP 1) Import the .csv file [labeled_data.csv](https://github.com/tpawelski/hate-speech-detection/blob/master/initial%20datasets/labeled_data.csv) (found in the 'initial datasets' folder)

STEP 2) Open the \'91feature engineering scripts\'92 folder and run \'91clean_tweets.py\'92 which performs initial text corpus pre-processing. The out-put of this script is \'91cleaned_tweets.csv\'92 which can also be found in the \'91initial datasets\'92 folder.\

STEP 3) Open and run each of the remaining scripts in the \'91feature engineering scripts\'92  folder which will create each of the feature subsets and output them as .csv files. The .csv file outputs from this step can be found in the \'91feature datasets\'92 folder. The dictionaries required for this step can be found in the \'91dictionaries\'92 folder. \

STEP 4) Open and run the script called \'91hate_speech_detection.py\'92 which reads in the .csv files in the \'91feature datasets\'92 folder, merges them into a single data frame, and trains models to classify instances as either hate speech, offensive language, or neither.\

STEP 5) Detect hate-speech and offensive language}
