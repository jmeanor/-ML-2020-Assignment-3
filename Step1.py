from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from pprint import pprint

import time
import numpy as np
import pandas as pd
# from sklearn.datasets import fetch_mldata
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class Step1():
    def __init__(self, data, name="", output="output"):
        self.data = data
        self.dataX, self.dataY = data['data'], data['target']
        self.name = name
    
    def run(self):
        # self.elbow(isKM=True)
        # self.silhouette_side_by_side(isKM=True)
        # self.silhouette(isKM=True)
        # self.visualize()

        # E.M.
        self.elbow(isKM=False)
        # self.silhouette_side_by_side(isKM=False)

    def k_means(self, k=5):
        km = KMeans(n_clusters=k, n_jobs=-1)
        clusters = km.fit_predict(self.dataX)

    # Analysis for selecting best number of clusters. 
    # Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    def silhouette(self, isKM=True):
        range_n_clusters = [2, 3, 4, 5, 6]

        for n_clusters in range_n_clusters:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            self._generate_silhouette(ax1, n_clusters)

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(self.dataX[:, 3], self.dataY, marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("Visualization of the clustered data.")
            ax2.set_xlabel("Feature space for 1st feature")
            ax2.set_ylabel("Feature space for 2nd feature")

            clustererType = 'K-Means' if isKM else 'E.M.'
            plt.suptitle(("Silhouette analysis for %s on %s "
                        "with n_clusters = %d" % (clustererType, self.name, n_clusters)),
                        fontsize=14, fontweight='bold')

        plt.show()

    # Analysis for selecting best number of clusters. 
    # Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    def silhouette_side_by_side(self, isKM):
        range_n_clusters = [2,6]

        for n_clusters in range_n_clusters:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            fig.set_size_inches(8, 8)

            self._generate_silhouette(ax1, n_clusters,   isKM)
            self._generate_silhouette(ax2, n_clusters+1, isKM)
            self._generate_silhouette(ax3, n_clusters+2, isKM)
            self._generate_silhouette(ax4, n_clusters+3, isKM)
            
            clustererType = 'K-Means' if isKM else 'E.M.'
            plt.suptitle(("Silhouette Analysis for %s on %s "% (clustererType, self.name)), fontweight='bold', fontsize=14)
            fig.subplots_adjust(wspace=0.5, hspace=0.5, left=0.125, right=0.9,top=0.9, bottom=0.1)
            plt.savefig('output/silhouette.png')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _generate_silhouette(self, ax1, n_clusters, isKM=True):
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(self.dataX) + (n_clusters + 1) * 10])
        
        if isKM:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        else:
            clusterer = GaussianMixture(n_components=n_clusters, init_params='random')

        data_df = pd.DataFrame(self.dataX)
        cluster_labels = clusterer.fit_predict(data_df)

        silhouette_avg = silhouette_score(data_df, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

        sample_silhouette_values = silhouette_samples(data_df, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Silhouette plot for %i clusters." %n_clusters)
        ax1.set_xlabel("Silhouette Coefficient")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    # TODO: Hardcoded for DS2
    def visualize(self):
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(self.dataX)

        feat_columns = ['month','day','temp_mean','temp_accum']
        data_df = pd.DataFrame(self.dataX, columns=feat_columns)
        data_df['y'] = self.dataY

        data_df['pca-one']   = pca_result[:, 0]
        data_df['pca-two']   = pca_result[:, 1]
        data_df['pca-three'] = pca_result[:, 2]
        pprint(data_df)

        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        ax = plt.figure(figsize=(8,5)).gca(projection='3d')
        ax.scatter(
            xs=data_df.loc[:,"pca-one"], 
            ys=data_df.loc[:,"pca-two"], 
            zs=data_df.loc[:,"pca-three"], 
            c= data_df.loc[:,"y"], 
            cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        plt.show()

    # Used to find optimal K
    # Source: https://heartbeat.fritz.ai/k-means-clustering-using-sklearn-and-python-4a054d67b187
    def elbow(self, isKM=True):
        Error =[]
        for i in range(1, 11):
            if isKM:
                km = KMeans(n_clusters = i, n_jobs=-1).fit(self.dataX)
                Error.append(km.inertia_)
            else:
                em = GaussianMixture(n_components=i, init_params='random').fit(self.dataX)
                Error.append((em.bic(self.dataX), em.aic(self.dataX)))
        import matplotlib.pyplot as plt
        plt.plot(range(1, 11), Error)
        clustererType = 'K-Means' if isKM else 'E.M.'
        plt.title('Elbow Method Analysis for %s on %s' %(clustererType, self.name))
        plt.xlabel('No of clusters')
        plt.ylabel('Error')
        plt.grid(True)
        plt.savefig('output/elbow.png')
        plt.show()
