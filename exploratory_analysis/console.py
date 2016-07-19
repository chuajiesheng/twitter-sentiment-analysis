import os
from utils import Reader
import code
import sys

if __name__ == '__main__':
    # coding=utf-8
    reload(sys)
    sys.setdefaultencoding('utf-8')

    working_directory = os.getcwd()
    files = Reader.read_directory(working_directory)
    print '{} available data files'.format(len(files))
    code.interact(local=dict(globals(), **locals()))
