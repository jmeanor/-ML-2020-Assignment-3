# Main
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint
import pandas as pd
# import errno
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
from Step1 import Step1

# Create output folder
# load_data.createDateFolder()

# Load Dataset
ds1 = load_data.loadDataset1()
ds2 = load_data.loadDataset2()


# Part 1
ds1_step1 = Step1(ds1)
ds1_step1.run()
ds2_step1 = Step1(ds2)
ds2_step1.run()

# Part 2
# Part 3
# Part 4
# Part 5