#!/bin/bash

set -x

if [ "$SNAP_PIPELINE_COUNTER" == "" ]; then export SNAP_PIPELINE_COUNTER=dev; fi
export PYTHONPATH=`pwd`

python3 -m nltk.downloader -d ~/nltk_data all
python3 analyzer/simple_naive_bayes.py
python3 analyzer/multinomial_naive_bayes.py
python3 analyzer/svm.py
python3 analyzer/vader.py
