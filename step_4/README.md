# Step 4 Sentiment Classification
Using the labelled data from the research assistants, we are going to create a a sentiment model to classify the tweets.
It will be a three class classification problem (negative, neutral, positive).

# Prelim result
This set of result is done with a 80% completed labels from the research assistants.

## Downsampling + RandomForestClassifier + LIWC values

- RandomForestClassifier
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad

```
[*] Average Train Accuracy/Error: 	0.983	0.017
[*] Average Test Accuracy/Error: 	0.654	0.346
[*] Average F1: 			0.651
```

Decent F1 but seems to have a gap between train and test error. 
Could be overfitting. Need to investigate more.

## Downsampling + RandomForestClassifier + LIWC values + Subjectivity score

- RandomForestClassifier
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad

```
[*] Average Train Accuracy/Error: 	0.982	0.018
[*] Average Test Accuracy/Error: 	0.674	0.326
[*] Average F1: 			0.671
```