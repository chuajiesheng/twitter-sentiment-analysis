# Step 3 Relevance Classification
Using the labelled data from the research assistants, we are going to create a a relevance model to classify the tweets.
It will be a two class classification problem.
If a tweet is relevant, it will be 1, else 0.

# Prelim result
This set of result is done with a 80% completed labels from the research assistants.

## Base

- LogisticRegression
- 100-best features using mutual information 
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic, Clout, Authentic, Tone, affect, posemo, negemo, anx, anger, sad

```
[*] Average Accuracy: 0.879
[*] Average Train Error: 0.121
[*] Average Test Error: 0.124
[*] Average F1: 0.755
[*] Average MCC: 0.562
```

This model performed so-so when trying to decide if it is not relevant (0-label) when given one.
Confusion matrix shows that it performed correctly 50% of the time.

## Base + more LIWC values

- LogisticRegression
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Accuracy: 0.879
[*] Average Train Error: 0.121
[*] Average Test Error: 0.127
[*] Average F1: 0.754
[*] Average MCC: 0.552
```

This model performed similar to the base model, except slightly better at determine if it is relevant when give one.
i.e. it say that a relevant is relevant (about 2 more than base) 

## MultinomialNB + more LIWC values

- MultinomialNB
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Accuracy: 0.649
[*] Average Train Error: 0.351
[*] Average Test Error: 0.363
[*] Average F1: 0.577
[*] Average MCC: 0.234
```

This model performed very well on deciding that it is relevant when given one.
Confusion matrix for one of the round:
```
[[650 395]
 [ 75 183]]
```
But it place too much irrelevant tweets to relevant, thus the low performance.

## SVM + more LIWC values

- SGDClassifier
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Accuracy: 0.610
[*] Average Train Error: 0.390
[*] Average Test Error: 0.389
[*] Average F1: 0.483
[*] Average MCC: 0.117
```

This model is not stable. It jumps between good and bad.
Example:
```
[[226 819]
 [ 42 216]]
 
[[1037    8]
 [ 253    5]]
```

## RandomForestClassifier + more LIWC values

- RandomForestClassifier
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Accuracy: 0.985
[*] Average Train Error: 0.015
[*] Average Test Error: 0.116
[*] Average F1: 0.786
[*] Average MCC: 0.598
```

This model give a generally high accuracy and slightly better MCC as compared to logistic regression.
It achieve this by a better deciding that a irrelevant is irrelevant.

The generally confusion matrix is this:
```
[[1014   31]
 [ 108  150]]
```

## RandomForestClassifier (more estimator) + more LIWC values

- RandomForestClassifier
    - n_estimators=50
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Accuracy: 0.997
[*] Average Train Error: 0.003
[*] Average Test Error: 0.109
[*] Average F1: 0.804
[*] Average MCC: 0.627
```

## Downsampling + RandomForestClassifier + more LIWC values

- RandomForestClassifier
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    -cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Accuracy: 0.986
[*] Average Train Error: 0.014
[*] Average Test Error: 0.229
[*] Average F1: 0.769
[*] Average MCC: 0.548
```

Test error is a bit high, it seems that the data from the training set is not fitted fully.

## Downsampling + RandomForestClassifier (more estimator) + more LIWC values

- RandomForestClassifier
    - n_estimators=50
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    -cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Accuracy: 0.996
[*] Average Train Error: 0.004
[*] Average Test Error: 0.198
[*] Average F1: 0.801
[*] Average MCC: 0.610
```

This classification is decent but it is starting to overfit with the test error and train error gap increaing to `0.19`.
But the gap with a 50 estimators seems to be lower that one with 10 estimators.
The overfitting started partly because the small training and testing size.

## Downsampling + RandomForestClassifier (even more estimator) + more LIWC values

- RandomForestClassifier
    - n_estimators=500
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    -cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Accuracy: 0.998
[*] Average Train Error: 0.002
[*] Average Test Error: 0.192
[*] Average F1: 0.807
[*] Average MCC: 0.621
```

The average MCC increased but the overfitting problem still exist.

## Downsampling + MLPClassifier (neural network) + more LIWC values

- MLPClassifier
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Train Accuracy/Error: 	0.856	0.144
[*] Average Test Accuracy/Error: 	0.747	0.253
[*] Average F1: 			0.745
[*] Average MCC: 			0.499
```
