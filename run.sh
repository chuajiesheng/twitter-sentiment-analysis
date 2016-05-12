#!/bin/bash

set -x

if [ "$SNAP_PIPELINE_COUNTER" == "" ]; then export SNAP_PIPELINE_COUNTER=dev; fi
export PYTHONPATH=`pwd`

python3 -m nltk.downloader -d ~/nltk_data all && \
python3 analyzer/simple_naive_bayes.py && \
python3 analyzer/multinomial_naive_bayes.py && \
python3 analyzer/svm.py && \
python3 analyzer/vader.py && \
true

pip2 install sklearn && \
python2 analyzer/sklearn_svm.py data/twitter_balanced > result/$SNAP_PIPELINE_COUNTER/sklearn_balanced.out && \
true

source activate neon && \
python2 analyzer/recurrent_neural_network.py -e 2 -eval 1 --rlayer_type lstm && \
source deactivate neon && \
true

