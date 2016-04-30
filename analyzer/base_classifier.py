#!/usr/bin/env python3

import logging
import os


class Classifier:
    FORMAT = '[%(asctime)s][%(levelname)-8s] #%(funcName)-10s → %(message)s'
    OUTPUT_FILE = 'result/{0}/{1}.out'


    logger = None

    def __init__(self):
        pass

    @staticmethod
    def init_logging():
        logging.basicConfig(format=Classifier.FORMAT, level=logging.DEBUG)

    @staticmethod
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def get_counter():
        return os.environ['SNAP_PIPELINE_COUNTER']

    @staticmethod
    def get_output_file(classifier_name):
        file = Classifier.OUTPUT_FILE.format(Classifier.get_counter(), classifier_name)
        Classifier.ensure_dir(file)
        return file
