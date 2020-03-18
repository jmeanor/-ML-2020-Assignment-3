import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pprint import pprint

import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
        # self.elbow()
        # inpt = input("Select your K value: ")
        # self.k_means(int(inpt))

        self.silhouette()
        # self.visualize()

    def k_means(self, k=5):
        km = KMeans(n_clusters=k)
        clusters = km.fit_predict(self.dataX)

    # Analysis for selecting best number of clusters. 
    # Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    def silhouette(self):
        range_n_clusters = [2, 3, 4, 5, 6]

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(self.dataX) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(self.dataX)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(self.dataX, cluster_labels)
            print("For n_clusters =", n_clusters,
                "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(self.dataX, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            ax1.set_xticks([])

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

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                        "with n_clusters = %d" % n_clusters),
                        fontsize=14, fontweight='bold')

        plt.show()

    # TODO: Hardcoded for DS2
    def visualize(self):
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(self.dataX)

        feat_columns = ['month','day','temp_mean','temp_accum']
        data_df = pd.DataFrame(self.dataX, columns=feat_columns)
        data_df['y'] = self.dataY

        # df = pd.DataFrame({'pca-one': pca_result[:, 0], 
        #     'pca-two': pca_result[:, 1], 
        #     'pca-three': pca_result[:, 2] }, columns=['pca-one', 'pca-two', 'pca-three'])
        data_df['pca-one']   = pca_result[:, 0]
        data_df['pca-two']   = pca_result[:, 1]
        data_df['pca-three'] = pca_result[:, 2]
        pprint(data_df)

        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        # plt.figure(figsize=(16,10))
        # sns.scatterplot(
        #     x="pca-one", y="pca-two",
        #     hue="y",
        #     palette=sns.color_palette("hls", 4),
        #     data=data_df,
        #     legend="full",
        #     alpha=0.3
        # )
        # plt.show()

        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
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