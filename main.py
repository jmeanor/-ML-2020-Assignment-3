# Main
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_files
from pprint import pprint
import pandas as pd
import graphviz
import errno
import os
from datetime import datetime
# Logging
import myLogger
import logging
logger = logging.getLogger()
logger.info('Initializing main.py')
log = logging.getLogger()
# Plotting
matplotlib.use("macOSX")
# Clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Assignment
import load_data

###
#  Creates an output directory and subdirectories. 
#  source: https://stackoverflow.com/questions/14115254/creating-a-folder-with-timestamp/14115286
###
def createDateFolder(suffix=("")):
    mydir = os.path.join(os.getcwd(), 'output', *suffix)
    # print('mydir %s' %mydir)
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return mydir

# Load Dataset
ds1 = load_data.loadDataset1()
ds2 = load_data.loadDataset2()

print('DS1', ds1, 'DS2', ds2)

# Part 1
# Part 2
# Part 3
# Part 4
# Part 5