# Step 3 Relevance Classification
Using the labelled data from the research assistants, we are going to create a a relevance model to classify the tweets.
It will be a two class classification problem.
If a tweet is relevant, it will be 1, else 0.

# Prelim result
This set of result is done with a 80% completed labels from the research assistants.

## Downsampling + Base

- LogisticRegression
- 100-best features using mutual information 
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic, Clout, Authentic, Tone, affect, posemo, negemo, anx, anger, sad

```
[*] Average Train Accuracy/Error: 	0.809	0.191
[*] Average Test Accuracy/Error: 	0.787	0.213
[*] Average F1: 			0.786
[*] Average MCC: 			0.576
```

This model performed so-so when trying to decide if it is not relevant (0-label) when given one.
Confusion matrix shows that it performed correctly 50% of the time.

## Downsampling + Base + more LIWC values

- LogisticRegression
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Train Accuracy/Error: 	0.808	0.192
[*] Average Test Accuracy/Error: 	0.773	0.227
[*] Average F1: 			0.773
[*] Average MCC: 			0.548
```

This model performed similar to the base model, except slightly better at determine if it is relevant when give one.
i.e. it say that a relevant is relevant (about 2 more than base) 

## Downsampling + MultinomialNB + more LIWC values

- MultinomialNB
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Train Accuracy/Error: 	0.654	0.346
[*] Average Test Accuracy/Error: 	0.652	0.348
[*] Average F1: 			0.652
[*] Average MCC: 			0.305
```

This model performed very well on deciding that it is relevant when given one.
Confusion matrix for one of the round:
```
[[650 395]
 [ 75 183]]
```
But it place too much irrelevant tweets to relevant, thus the low performance.

## Downsampling + SVM + more LIWC values

- SGDClassifier
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Train Accuracy/Error: 	0.558	0.442
[*] Average Test Accuracy/Error: 	0.552	0.448
[*] Average F1: 			0.476
[*] Average MCC: 			0.136
```

This model is not stable. It jumps between good and bad.
Example:
```
[[ 40 218]
 [ 14 244]]
 
[[256   2]
 [255   3]]
```

## Downsampling + RandomForestClassifier + more LIWC values

- RandomForestClassifier
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Train Accuracy/Error: 	0.985	0.015
[*] Average Test Accuracy/Error: 	0.764	0.236
[*] Average F1: 			0.762
[*] Average MCC: 			0.536
```

Test error is a bit high, it seems that the data from the training set is not fitted fully.

## Downsampling + RandomForestClassifier (more estimator) + more LIWC values

- RandomForestClassifier
    - n_estimators=50
- 100-best features using mutual information
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Train Accuracy/Error: 	0.997	0.003
[*] Average Test Accuracy/Error: 	0.797	0.203
[*] Average F1: 			0.796
[*] Average MCC: 			0.602
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
    - cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Train Accuracy/Error: 	0.997	0.003
[*] Average Test Accuracy/Error: 	0.821	0.179
[*] Average F1: 			0.821
[*] Average MCC: 			0.648
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
