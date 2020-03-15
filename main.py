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

# Create output folder
# load_data.createDateFolder()

# Load Dataset
ds1 = load_data.loadDataset1()
ds2 = load_data.loadDataset2()


# Part 1
# Part 2
# Part 3
# Part 4
# Part 5