import h5py as h5

from bokeh.layouts import column
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

import random
import time
import os
import logging

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def findData():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, '..', 'tmp')
    files = list(absoluteFilePaths(data_path))
    files = [f for f in files if f.endswith(".h5")]
    if len(files) <= 0:
        logging.error("no files in data directory")
        logging.error("aborting...")
        return None
    most_recent = max(files, key=os.path.getctime)
    return most_recent

def update():
    pass

def main():
    data = findData()
    

if __name__ == "__main__":
    main()
