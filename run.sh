#!/bin/bash

export PYTHONPATH=`pwd`
python3 analyzer/simple_naive_bayes.py
python3 analyzer/vader.py
