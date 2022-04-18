#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from common import describe_data, test_env
__all__ = [describe_data, test_env]


def plot_clusters(X, y, figure, file):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:olive']
    markers = ['o', 'X', 's', 'D']
    color_idx = 0
    marker_idx = 0

    plt.figure(figure)

    for cluster in range(0, len(set(y))):
        plt.scatter(X[y == cluster, 0], X[y == cluster, 1],
                    s=5, c=colors[color_idx], marker=markers[marker_idx])
        color_idx = 0 if color_idx == (len(colors) - 1) else color_idx + 1
        marker_idx = 0 if marker_idx == (len(markers) - 1) else marker_idx + 1

    plt.title(figure)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(file, papertype='a4')

    plt.show()


def clustering(df):
    Sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters=k, init='k-means++', random_state=0)
        km = km.fit(df)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances)
    plt.xlabel('number of clusters')
    plt.ylabel('WCSS')
    plt.grid()
    plt.title('The Elbow Method')
    plt.savefig('results/sd_wcss_plot.png', papertype='a4')
    plt.show()


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)
    df = pd.read_csv('data/TTWO.csv')
    df = df.fillna(df.mean())
    X = df.values
    clustering(X)
    n_clusters = 8
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    y_kmeans = k_means.fit_predict(X)
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)

    describe_data.print_overview(
        df, file='results/sd_overview.txt')

    plot_clusters(X_tsne, np.full(X_tsne.shape[0], 0),
                  't-SNE visualisation without clusters', 'results/sd_tsne_no_clusters.png')
    plot_clusters(X_tsne, y_kmeans, 'k means clusters with TSNE',
                  'results/sd_tsne_X_clusters.png')
