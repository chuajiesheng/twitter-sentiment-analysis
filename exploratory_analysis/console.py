import os
from utils import Reader
import code

if __name__ == '__main__':
    working_directory = os.getcwd()
    files = Reader.read_directory(working_directory)
    print '{} available data files'.format(len(files))
    code.interact(local=dict(globals(), **locals()))
