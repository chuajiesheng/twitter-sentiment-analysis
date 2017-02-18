# Step 3 Relevance Classification
Using the labelled data from the research assistants, we are going to create a a relevance model to classify the tweets.
It will be a two class classification problem.
If a tweet is relevant, it will be 1, else 0.

# Prelim result
This set of result is done with a 80% completed labels from the research assistants.

## Base

- LogisticRegression
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    -cross_validation=10
- LIWC values: Analytic, Clout, Authentic, Tone, affect, posemo, negemo, anx, anger, sad

```
[*] Average Accuracy: 0.879
[*] Average Train Error: 0.121
[*] Average Test Error: 0.124
[*] Average F1: 0.755
[*] Average MCC: 0.562
```

## Base + more LWIC values

- LogisticRegression
- StratifiedShuffleSplit
    - train=80%
    - test=20%
    -cross_validation=10
- LIWC values: Analytic,Clout,Authentic,Tone,affect,posemo,negemo,anx,anger,sad,social,family,friend,female,male,percept,see,hear,feel,focuspast,focuspresent,focusfuture,relativ,motion,space,time,work,leisure,home,money,relig,death

```
[*] Average Accuracy: 0.879
[*] Average Train Error: 0.121
[*] Average Test Error: 0.127
[*] Average F1: 0.754
[*] Average MCC: 0.552
```
