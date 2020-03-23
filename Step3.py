from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pprint import pprint

import time, os
import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
import logging
log = logging.getLogger()
import load_data
from Step1 import Step1

class Step3():
    algs = [PCA, FastICA, GaussianRandomProjection]
    markers = ["*", "d", "<", ">"]
    ls = ['solid', 'dotted', 'dashed', 'dashdot']

    def __init__(self, data, name="", output="output", k=[]):
        self.data = data
        self.dataX, self.dataY = data['data'], data['target']
        self.name = name
        self.output=output
        self.range = range(1, self.dataX.shape[1]+1)
        self.seeds=[ 7, 21, 95, 108, 1053]
        self.n_components=k
        self.cluster_range= range(100,1000,200) if self.name == "AirBNB-DS" else range(1,11)
    
    def run(self):
        self.output = load_data.createDateFolder(suffix=["Step3"])
        for (idx, alg) in enumerate(self.algs): 
            X_transform = self.dim_reduc(alg, self.n_components[idx])
            data = {
                'data': X_transform,
                'target': self.dataY
            }
            step1 = Step1(data, name=self.name + alg.__name__, output=self.output, cluster_range=self.cluster_range)
            step1.run()
            

    def dim_reduc(self, Algorithm, k):
        if Algorithm == LinearDiscriminantAnalysis:
            X_train, X_test, y_train, y_test = train_test_split(
                self.dataX, self.dataY, test_size=0.2, random_state=0)
            rp = Algorithm(n_components=k)
            rp.fit(X_train, y_train)
            X = rp.transform(X_test)
        else:
            rp = Algorithm(n_components=k, random_state=10)
            rp.fit(self.dataX)
            X = rp.transform(self.dataX)
        return X