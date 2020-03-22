from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
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
    def __init__(self, data, name="", output="output"):
        self.data = data
        self.dataX, self.dataY = data['data'], data['target']
        self.name = name
        self.output=output
        self.range = range(1, self.dataX.shape[1]+1)
        self.seeds=[ 7, 21, 95, 108, 1053]
    
    def run(self):
        self.output = load_data.createDateFolder(suffix=["Step2"])
        # self.PCA()
        # self.ICA()
        # self.RP()

        # Refactor
        algs = [PCA, FastICA, GaussianRandomProjection]
        errs = []
        kurts = []

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Dimensionality Reduction')
        ax1.set(title='Avg Reconstruction MSE', xlabel='k', ylabel='Error')
        ax1.grid(True)

        ax2.set(title='Avg Kurtoses', xlabel='k', ylabel='Kurtosis')
        ax2.grid(True)
        # ax1.set_xlim()
        

        for alg in algs:
            err, kurt = self.dim_reduc(Algorithm=alg)
            # err = {alg.__name__: err}
            # kurt = {alg.__name__: kurt}
            errs.append(err)
            kurts.append(kurt)
            ax1.plot(self.range, err.loc[:,'loss'], label=alg.__name__)
            ax2.plot(self.range, kurt.loc[:,'kurtosis'], label=alg.__name__)
        # plt.clf()
        # errs = pd.DataFrame(errs)
        # kurts = pd.DataFrame(kurts)
        # plt.plot(errs, label="Avg Reconstruction Error")
        # plt.plot(kurts, label="Avg Kurtosis")
        plt.grid(True)
        ax1.legend(loc="best")
        ax2.legend(loc="best")
        plt.show()
        plt.savefig(os.path.join(self.output, self.name + '.png'))
        # self.dim_reduc(Algorithm=PCA)
        # self.dim_reduc(Algorithm=FastICA)
        # self.dim_reduc(Algorithm=GaussianRandomProjection)
        
    
    # Source: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
    def PCA(self):
        best_loss, best_k = 1000, 0
        errors = []
        kurtoses = []
        for k in range(1, self.dataX.shape[1] + 1):
            losses = []
            kurts = []
            for r in self.seeds:
                pca = PCA(n_components=k, random_state=r)
                pca.fit(self.dataX)
                X = pca.transform(self.dataX)
                y = self.dataY

                X_projected = pca.inverse_transform(X)
                # log.info('X_projected: %s' %X_projected)
                loss = ((self.dataX - X_projected) ** 2).mean().mean()
                losses.append(loss)
                kurts.append(scipy.stats.kurtosis(X) ** 2)
            avgLoss = np.mean(losses)
            avgKurt = np.mean(kurts)
            err = {'n_components': k, 'loss': avgLoss}
            errors.append(err)
            kurtoses.append({'n_components': k, 'kurtosis^2': avgKurt})
            # log.info('PCA Reconstruction Error: %f' %loss)
            if avgLoss < best_loss:
                best_loss = avgLoss
                best_k = k
        errors = pd.DataFrame(errors)
        kurtoses = pd.DataFrame(kurtoses)
        plt.plot(range(1, self.dataX.shape[1]+1), errors.loc[:,'loss'])
        plt.plot(range(1, self.dataX.shape[1]+1), kurtoses.loc[:,'kurtosis^2'])
        plt.show()
        log.info(errors)
        log.info('PCA\tBest n_components found: %i with avg loss: %f' %(best_k, best_loss) )

        # Logging
        # log.info('Before PCA:')
        # log.info(self.dataX)
        # log.info(self.dataX.shape) 
        # pd.DataFrame(self.dataX).to_csv(os.path.join(self.output, self.name + "_before.csv"))
        pd.DataFrame(X).to_csv(os.path.join(self.output, self.name+"_after.csv"))
        # log.info('After PCA:')
        # log.info(X)
        # log.info(X.shape)

        # Plot in 3D Space
        # fig = plt.figure(1, figsize=(4, 3))
        # plt.clf()
        # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        # for name, label in [('Bud', 1), ('Partial', 2), ('Bloom', 3), ('Full', 4)]:
        #     ax.text3D(X[y == label, 0].mean(),
        #             X[y == label, 1].mean() + 1.5,
        #             X[y == label, 2].mean(), name,
        #             horizontalalignment='center',
        #             bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        # y = np.choose(y, [0, 1, 2, 3, 4]).astype(np.float)
        # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
        #         edgecolor='k')
        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])

        # plt.show()

    def ICA(self):
        best_loss, best_k = 1000, 0
        errors = []
        for k in range(1, self.dataX.shape[1]+1 ):
            losses = []
            for r in self.seeds:
                ica = FastICA(n_components=k, random_state=r)
                ica.fit(self.dataX)
                X = ica.transform(self.dataX)
                y = self.dataY

                X_projected = ica.inverse_transform(X)
                # log.info('X_projected: %s' %X_projected)
                loss = ((self.dataX - X_projected) ** 2).mean().mean()
                losses.append(loss)
            avgLoss = np.mean(losses)
            err = {'n_components': k, 'loss': avgLoss}
            errors.append(err)
            # log.info('ICA Reconstruction Error: %f' %loss)
            if avgLoss < best_loss:
                best_loss = avgLoss
                best_k = k
        errors = pd.DataFrame(errors)
        plt.plot(range(1, self.dataX.shape[1]+1), errors.loc[:,'loss'])
        plt.show()
        log.info(errors)
        log.info('ICA\tBest n_components found: %i with avg loss: %f' %(best_k, best_loss) )

        # log.info('ICA Reconstruction Error: %f' %loss)

        # Plot
        # fig = plt.figure(1, figsize=(4, 3))
        # plt.clf()
        # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        # for name, label in [('Bud', 1), ('Partial', 2), ('Bloom', 3), ('Full', 4)]:
        #     ax.text3D(X[y == label, 0].mean(),
        #             X[y == label, 1].mean() + 1.5,
        #             X[y == label, 2].mean(), name,
        #             horizontalalignment='center',
        #             bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

        # Reorder the labels to have colors matching the cluster results
        # y = np.choose(y, [0, 1, 2, 3, 4]).astype(np.float)
        # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
        #         edgecolor='k')

        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])

        # plt.show()

    def RP(self):
        best_loss, best_k = 1000, 0
        errors = []
        for k in range(1, self.dataX.shape[1]+1):
            losses = []
            for r in self.seeds:
                rp = GaussianRandomProjection(n_components=k, random_state=r)
                rp.fit(self.dataX)
                X = rp.transform(self.dataX)
                y = self.dataY

                # Measure avg reconstruction error
                X_projected = np.dot(X, np.linalg.pinv(rp.components_.T))
                loss = ((self.dataX - X_projected) ** 2).mean().mean()
                losses.append(loss)
            avgLoss = np.mean(losses)
            err = {'n_components': k, 'loss': avgLoss}
            errors.append(err)
            # log.info('RP  Reconstruction Error: %f' %loss)
            if avgLoss < best_loss:
                best_loss = avgLoss
                best_k = k
        errors = pd.DataFrame(errors)
        plt.clf()
        plt.plot(range(1, self.dataX.shape[1]+1), errors.loc[:,'loss'])
        plt.show()
        log.info(errors)
        log.info('RP\tBest n_components found: %i with avg loss: %f' %(best_k, best_loss) )

        # log.info('RP Reconstruction Error: %f' %loss)

        # fig = plt.figure(1, figsize=(4, 3))
        # plt.clf()
        # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        # for name, label in [('Bud', 1), ('Partial', 2), ('Bloom', 3), ('Full', 4)]:
        #     ax.text3D(X[y == label, 0].mean(),
        #             X[y == label, 1].mean() + 1.5,
        #             X[y == label, 2].mean(), name,
        #             horizontalalignment='center',
        #             bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

        # Reorder the labels to have colors matching the cluster results
        # y = np.choose(y, [0, 1, 2, 3, 4]).astype(np.float)
        # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
        #         edgecolor='k')

        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])

        # plt.show()

    def dim_reduc(self, Algorithm):
        best_loss, best_k = 1000, 0
        kurtoses = []
        errors = []
        for k in self.range:
            kurts = []
            losses = []
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

        log.info(errors)
        log.info('%s\tBest n_components found: %i with avg loss: %f' %(Algorithm.__name__, best_k, best_loss))

        return (errors, kurtoses)