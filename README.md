# Sentiment Analysis for Twitter Data

## Dataset

```
Total records: 10000
Total unique records: 9951
```

### Training and Testing

```
Training records: 3,317
Total unique records: 6,634
```

## Naive Bayes with Unigram Features

### Features

1. Negated words
2. Unigram features with minimum frequency of 4
3. Naive Bayes Classifier

### Results

```
Accuracy: 0.55712993668978
F-measure [negative]: 0.34994807892004154
F-measure [neutral]: 0.42481751824817515
F-measure [positive]: 0.6875
Precision [negative]: 0.39369158878504673
Precision [neutral]: 0.4855394883203559
Precision [positive]: 0.6246231155778894
Recall [negative]: 0.3149532710280374
Recall [neutral]: 0.37759515570934254
Recall [positive]: 0.7644526445264452
```

### Note

This method have been replaced with *Naive Bayes with Unigram and Bigram Features*.

## Vader Classification

### Features

1. Sentiment Intensity Analyzer
2. Obtain polarity scores for each text
3. Calculate accuracy

### Results

```
Total correct: 3528
Total wrong: 6423
Total accuracy: 0.35453723243895086
```

## Naive Bayes with Unigram and Bigram Features

### Features

1. Negated words
2. Unigram features with minimum frequency of 4
3. Bigram features without minimum frequency
4. Naive Bayes Classifier

### Results

```
Accuracy: 0.5482363581549593
F-measure [negative]: 0.34303864478560087
F-measure [neutral]: 0.42152046783625735
F-measure [positive]: 0.6790540540540541
Precision [negative]: 0.3927272727272727
Precision [neutral]: 0.45072536268134067
Precision [positive]: 0.6330708661417322
Recall [negative]: 0.30451127819548873
Recall [neutral]: 0.3958699472759227
Recall [positive]: 0.73224043715847
```

