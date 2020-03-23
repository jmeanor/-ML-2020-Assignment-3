from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
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

class Step2():
    algs = [PCA, FastICA, GaussianRandomProjection, LinearDiscriminantAnalysis]
    # markers = ["*", "d", "<", ">"]
    markers = [".", ".", ".", "."]
    ls = ['solid', 'dotted', 'dashed', 'dashdot']

    def __init__(self, data, name="", output="output"):
        self.data = data
        self.dataX, self.dataY = data['data'], data['target']
        self.name = name
        self.output=output
        self.range = range(1, self.dataX.shape[1]+1)
        self.seeds=[ 7, 21, 95, 108, 1053]
    
    def run(self):
        self.output = load_data.createDateFolder(suffix=["Step2"])
        self.find_optimums()

    def find_optimums(self):
        # Refactor
        errs = []
        kurts = []

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(16, 6)
        fig.suptitle('Dimensionality Reduction')

        ax1.set(title='PCA Explained Variance')
        self.plot_EV(ax1)

        ax2.set(title='Avg Reconstruction MSE', xlabel='k', ylabel='Error')
        ax2.grid(True)

        ax3.set(title='Avg Kurtoses', xlabel='k', ylabel='Kurtosis')
        ax3.grid(True)
        
        for i, alg in enumerate(self.algs):
            err, kurt = self.dim_reduc(Algorithm=alg)
            errs.append(err)
            kurts.append(kurt)
            if alg != LinearDiscriminantAnalysis:
                ax2.plot(self.range, err.loc[:,'loss'], label=alg.__name__, marker=self.markers[i], ls=self.ls[i])
                log.info('%s\t\tMin Reconstruction Error Index: %i, MSE: %f' %(alg.__name__,err.loc[:,'loss'].idxmin(), err.loc[:,'loss'].min()))
            ax3.plot(self.range, kurt.loc[:,'kurtosis'], label=alg.__name__, marker=self.markers[i], ls=self.ls[i])
            log.info('%s\t\tMax Kurtosis Index: %i, Kurtosis: %f' %(alg.__name__,kurt.loc[:,'kurtosis'].idxmax(), kurt.loc[:,'kurtosis'].max()))
        plt.grid(True)
        ax2.legend(loc="best")
        ax3.legend(loc="best")
        plt.savefig(os.path.join(self.output, self.name + '.png'))
        # plt.show()
        
        # Select the best
        # for i, alg in enumerate(self.algs):
    def runLDA(self): 
        # scores = []
        # for k in self.range:
        #     X_train, X_test, y_train, y_test = train_test_split(
        #         self.dataX, self.dataY, test_size=0.2, random_state=0)
        #     rp = LinearDiscriminantAnalysis(n_components=k)
        #     rp.fit(X_train, y_train)
        #     X = rp.transform(X_test)
        #     y = self.dataY
        #     scores.append(rp.explained_variance_ratio_)

        fig, (axis) = plt.subplots(1, 1)
        fig.set_size_inches(6, 6)
        fig.suptitle('Dimensionality Reduction')
        axis.set(title='LDA Explained Variance')
        axis.grid(True)

        X_train, X_test, y_train, y_test = train_test_split(
                self.dataX, self.dataY, test_size=0.5, random_state=0)
        rp = LinearDiscriminantAnalysis(store_covariance=True)
        rp.fit(X_train, y_train)
        X = rp.transform(X_test)
        y = self.dataY

        cov_mat = rp.covariance_
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        total = len(rp.explained_variance_ratio_)
        var_exp = [(i / total) for i in sorted(eigen_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)

        # plot explained variances
        axis.bar(self.range, var_exp, alpha=0.5,
                align='center', label='Individual Explained Variance')
        axis.step(self.range, cum_var_exp, where='mid',
                label='Cumulative Explained Variance')
        axis.set(ylabel='Explained Variance atio', xlabel='k')
        axis.legend(loc='best')
        axis.grid(True)
        plt.savefig(os.path.join(self.output, self.name + '-explained-variance.png'))
        # plt.show()
        
    # Explained Variance
    # Source: https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad
    def plot_EV(self, axis):
        cov_mat = np.cov(self.dataX.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        total = sum(eigen_vals)
        var_exp = [(i / total) for i in sorted(eigen_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)

        # plot explained variances
        axis.bar(self.range, var_exp, alpha=0.5,
                align='center', label='Individual Explained Variance')
        axis.step(self.range, cum_var_exp, where='mid',
                label='Cumulative Explained Variance')
        axis.set(ylabel='Explained variance ratio', xlabel='k')
        axis.legend(loc='best')
        axis.grid(True)
        return axis
        # plt.show()
    
    # Source: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html

    def dim_reduc(self, Algorithm):
        best_loss, best_k = 1000, 0
        kurtoses = []
        errors = []
        for k in self.range:
            kurts = []
            losses = []

            if Algorithm == LinearDiscriminantAnalysis:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.dataX, self.dataY, test_size=0.2, random_state=0)
                rp = Algorithm(n_components=k)
                rp.fit(X_train, y_train)
                X = rp.transform(X_test)
                y = self.dataY

                kurts.append(np.abs(scipy.stats.kurtosis(X)))
            else: 
                for r in self.seeds:
                    rp = Algorithm(n_components=k, random_state=r)
                    rp.fit(self.dataX)
                    X = rp.transform(self.dataX)
                    y = self.dataY

                    # Measure avg reconstruction error
                    X_projected = np.dot(X, np.linalg.pinv(rp.components_.T))
                    loss = ((self.dataX - X_projected) ** 2).mean().mean()

                    losses.append(loss)
                    kurts.append(np.abs(scipy.stats.kurtosis(X)))
            avgLoss = np.mean(losses)
            avgKurt = np.mean(kurts)
            err = {'n_components': k, 'loss': avgLoss}
            errors.append(err)
            kurtoses.append({'n_components': k, 'kurtosis': avgKurt})
            # log.info('%s  Reconstruction Error: %f' %(Algorithm.__name__,loss))
            if avgLoss < best_loss:
                best_loss = avgLoss
                best_k = k
        errors = pd.DataFrame(errors)
        kurtoses = pd.DataFrame(kurtoses)
        
        # plt.clf()
        # plt.plot(self.range, errors.loc[:,'loss'], label="Reconstruction Error")
        # plt.title("Reconstruction Error for %s, %s" %(Algorithm.__name__, self.name))
        # plt.grid(True)
        # plt.legend(loc="best")
        # plt.savefig(os.path.join(self.output, "%s-%s-recon-error.png" %(self.name, Algorithm.__name__)))
        # plt.show()

        # plt.clf()
        # plt.plot(self.range, kurtoses.loc[:,'kurtosis'], label="Kurtosis")
        # plt.grid(True)
        # plt.title("Kurtoses for %s, %s" %(Algorithm.__name__, self.name))
        # plt.savefig(os.path.join(self.output, "%s-%s-kurtosis.png" %(self.name, Algorithm.__name__)))
        # plt.show()

        # log.info(errors)
        # log.info('%s\tBest n_components found: %i with avg loss: %f' %(Algorithm.__name__, best_k, best_loss))

        return (errors, kurtoses)