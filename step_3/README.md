# Step 3 Relevance Classification
Using the labelled data from the research assistants, we are going to create a a relevance model to classify the tweets.
It will be a two class classification problem.
If a tweet is relevant, it will be 1, else 0.

## Methodology

We first investigate how the model will performed using a baseline method, Naive Bayes.

### Baseline

Experiment setup:

    - Naive Bayers classifier (with default values)
    - 10-fold cross validation
    - Stratified shuffle split (same number of each class per split)
    - 80% training data, 20% testing data
    - Treebank tokenzier

Result:

```
[*] Average Train Accuracy/Error: 	0.881	0.119
[*] Average Test Accuracy/Error: 	0.811	0.189
[*] Average F1: 			0.811
[*] Average MCC: 			0.623
```