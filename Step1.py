import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pprint import pprint


# 
# from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class Step1():
    def __init__(self, data):
        self.data = data
        self.dataX, self.dataY = data['data'], data['target']
    
    def run(self):
        self.elbow()
        inpt = input("Select your K value: ")
        self.k_means(int(inpt))
    
    def k_means(self, k=5):
        km = KMeans(n_clusters=k)
        clusters = km.fit_predict(self.dataX)


    # Used to find optimal K
    # Source: https://heartbeat.fritz.ai/k-means-clustering-using-sklearn-and-python-4a054d67b187
    def elbow(self):
        Error =[]
        for i in range(1, 11):
            km = KMeans(n_clusters = i).fit(self.dataX)
            # km.fit(self.dataX)
            Error.append(km.inertia_)
        import matplotlib.pyplot as plt
        plt.plot(range(1, 11), Error)
        plt.title('Elbow method')
        plt.xlabel('No of clusters')
        plt.ylabel('Error')
        plt.show()