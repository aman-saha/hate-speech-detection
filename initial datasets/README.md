# Labeled Dataset
The dataset used for this project contains 24,784 tweets manually labeled by CrowdFlower users as  as `hate_speech`, `offensive_language`, or `neither`.

## Definitions
`index`: the unique identifier of the tweet

`count`: the total # of CrowdFlower users who labeled the tweet 

`hate_speech`: the # of CrowdFlower users who labeled the tweet as containing or constituting hate speech

`offensive_language`: the # of CrowdFlower users who labeled the tweet as containing or constituting offensive language 

`neither`: the # of CrowdFlower users who labeled the Tweet as neither hate speech nor offensive language

`class`: the majority label given by CrowdFlower users (0 represents hate speech, 1 represents offensive language, and 2 represents neither)

`tweet`: the tweet, in textual form 

`clean_tweet`: the text of the tweet after removing punctuation and converting to lower-case

## Data Citation
The dataset mentioned above was obtained from:

Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language." Proceedings of the 11th International Conference on Web and Social Media (ICWSM). 
