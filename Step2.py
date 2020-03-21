from sklearn.decomposition import PCA, FastICA
from pprint import pprint

import time, os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
import logging
log = logging.getLogger()

class Step2():
    def __init__(self, data, name="", output="output"):
        self.data = data
        self.dataX, self.dataY = data['data'], data['target']
        self.name = name
        self.output=output
    
    def run(self):
        self.PCA()
        # self.ICA()
    
    # Source: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
    def PCA(self):
        pca = PCA(n_components=3)
        pca.fit(self.dataX)
        X = pca.transform(self.dataX)
        y = self.dataY

        log.info('Before PCA:')
        # log.info(self.dataX)
        log.info(self.dataX.shape) 
        # pd.DataFrame(self.dataX).to_csv(os.path.join(self.output, self.name + "_before.csv"))
        pd.DataFrame(X).to_csv(os.path.join(self.output, self.name+"_after.csv"))
        log.info('After PCA:')
        # log.info(X)
        log.info(X.shape)

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        for name, label in [('Bud', 1), ('Partial', 2), ('Bloom', 3), ('Full', 4)]:
            ax.text3D(X[y == label, 0].mean(),
                    X[y == label, 1].mean() + 1.5,
                    X[y == label, 2].mean(), name,
                    horizontalalignment='center',
                    bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        # y = np.choose(y, [1, 2, 0]).astype(np.float)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
                edgecolor='k')

        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])

        plt.show()

    def ICA(self):
        pca = FastICA(n_components=3)
        pca.fit(self.dataX)
        X = pca.transform(self.dataX)
        y = self.dataY

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        for name, label in [('Bud', 1), ('Partial', 2), ('Bloom', 3), ('Full', 4)]:
            ax.text3D(X[y == label, 0].mean(),
                    X[y == label, 1].mean() + 1.5,
                    X[y == label, 2].mean(), name,
                    horizontalalignment='center',
                    bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 2, 0]).astype(np.float)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
                edgecolor='k')

        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])

        plt.show()