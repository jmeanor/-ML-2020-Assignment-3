# Main
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint
import pandas as pd
import seaborn as sns
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
import analysis as an
from Step1 import Step1
from Step2 import Step2
from Step3 import Step3

# Create output folder
output_dir = load_data.createDateFolder()
load_data.setLog(output_dir)

# Load Dataset
ds1 = load_data.loadDataset1()
ds2 = load_data.loadDataset2()

# Scale the data sets
ds1 = load_data.normalize_features(ds1)
ds2 = load_data.normalize_features(ds2)

# pprint(ds1)
# pprint(ds2)

# Analysis
# an.heatmap(ds1, output_dir, 'AirBNB-DS', 'Blues')
# an.heatmap(ds2, output_dir, 'CherryBl-DS', 'Reds')

# Part 1
# ds1_step1 = Step1(ds1, name="AirBNB-DS", cluster_range=range(100,1000,200))
# ds1_step1.run()
# ds2_step1 = Step1(ds2, name="Cherry-Blossom-DS")
# ds2_step1.run()

# Part 2
ds1_step2 = Step2(ds1, name="AirBNB-DS", output=output_dir)
ds1_step2.run()
# ds2_step2 = Step2(ds2, name="Cherry Blossom DS", output=output_dir)
# ds2_step2.run()

# Manual
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ds1_step2.plot_EV(ax)
# plt.show()
# ds1_step2.runLDA()
# ds2_step2.runLDA()

# Part 3
# ds1_step3 = Step3(ds1, name="AirBNB-DS", output=output_dir, k=[20, 52, 30, 30])
# ds1_step3.run()
# ds2_step3 = Step3(ds1, name="Cherry Blossom DS", output=output_dir, k=[2, 3, 3, 2])
# ds2_step3.run()

# Part 4
# Part 5

print('Done')