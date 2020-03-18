# Copied from my Assignmnet 1 deliverable.
# Source: https://github.com/jmeanor/ML-2020

# Main
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_files
import pandas as pd
import errno
import os
from datetime import datetime
# Logging
import myLogger
import logging
logger = logging.getLogger()
logger.info('Initializing main.py')
log = logging.getLogger()


def setLog(path, oldHandler = None):
    if oldHandler != None:
        myLogger.logger.removeHandler(oldHandler)
    logPath = os.path.join(path, 'metadata.txt')
    fh = myLogger.logging.FileHandler(logPath)
    fh.setLevel(logging.INFO)
    fmtr = logging.Formatter('%(message)s')
    fh.setFormatter(fmtr)
    myLogger.logger.addHandler(fh)
    return fh

# ==========================================
#   Load Data Set 1
# ==========================================
def loadDataset1():
    df = pd.read_csv('./input/singapore-listings.csv', delimiter=',', header=0)
    
    # Pre-Processing - Removing attributes with no value
    df = df.drop(['id', 'name', 'host_name', 'last_review'], axis=1)

    # ==========================================
    # Discretize the classifications 
    #  Source: https://dfrieds.com/data-analysis/bin-values-python-pandas
    # ==========================================
    df['price_bins'] = pd.cut(x=df['price'], bins= np.arange(10, 10010, step=10), labels=np.arange(10, 10000, step=10)).astype(int)


    # ==========================================
    # PreProcessing - Encoding Categories
    # Source - https://blog.cambridgespark.com/robust-one-hot-encoding-in-python-3e29bfcec77e
    # ==========================================
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.preprocessing import KBinsDiscretizer
    categories = [
        'neighbourhood_group',
        'neighbourhood',
        'room_type'
    ]
    df_processed = pd.get_dummies(df, prefix_sep="__", columns=categories)
    dummies = [col for col in df_processed
        if "__" in col and col.split("__")[0] in categories]
    processed_columns = list(df_processed.columns[:])

    # print(processed_columns)
    # target = np.array(df_processed['price'])
    # data = np.array(df_processed.drop('price', axis=1))
    target = np.array(df_processed['price_bins'])
    data = np.array(df_processed.drop('price_bins', axis=1))
    
    # 2nd Iteration with imputing missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value=0)
    imputer = imputer.fit(data)
    data = imputer.transform(data)
    # =====================================
    
    data1 = {
        'data': data,
        'target': target
    }
    # log.debug(data1)
    return data1


# ==========================================
#   Load Data Set 2
# ==========================================
def loadDataset2():
    df = pd.read_csv('./input/hirosaki_temp_cherry.csv', delimiter=',', header=0)

    target = np.array(df['flower_status'])
    data = np.array(df.drop('flower_status', axis=1))
    data2 = {
        'data': data,
        'target': target
    }
    return data2

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