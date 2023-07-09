import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

def cure(data, n_clusters, alpha, verbose=True):
    # Step 1: Initialize clusters with all data points
    clusters = [{i} for i in range(len(data))]
    num_clusters = len(clusters)
    if verbose:
        print(f"Initial number of clusters: {num_clusters}")

    # Step 2: Repeat until the number of clusters is less than or equal to n_clusters
    while num_clusters > n_clusters:
        # Step 3: Select the medoids
        num_medoids = max(int(alpha * len(data)), n_clusters)
        medoids = []
        for i in range(num_medoids):
            max_dist = -1
            medoid = None
            for c in clusters:
                if len(c) == 1:
                    dists = [0]
                else:
                    cluster_data = data[list(c)]
                    centroid = np.mean(cluster_data[:, 1:], axis=0)
                    dists = np.sum((cluster_data[:, 1:] - centroid)**2, axis=1)
                if max(dists) > max_dist:
                    max_dist = max(dists)
                    medoid = list(c)[np.argmax(dists)]
            medoids.append(medoid)

        # Step 4: Cluster the medoids using Agglomerative Clustering
        medoid_data = data[medoids][:, 1:]
        if not medoids:  # add this check
          continue
        agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        agglomerative_clustering.fit(medoid_data)

        # Step 5: Merge the clusters whose medoids were grouped together in the Agglomerative Clustering
        new_clusters = [set() for i in range(n_clusters)]
        for i, c in enumerate(agglomerative_clustering.labels_):
            print("i:", i, "c:", c, "medoids[i]:", medoids[i])
            new_clusters[c].update(clusters[medoids[i]])
        clusters = [c for c in clusters if c not in [set(), set(medoids)]]
        clusters += new_clusters
        num_clusters = len(clusters)
        if verbose:
            print(f"Number of clusters after merging: {num_clusters}")

        # Step 6: Shrink the size of the clusters
        for i in range(len(clusters)):
            if len(clusters[i]) > 1:
                cluster = data[list(clusters[i])]
                centroid = np.mean(cluster[:, 1:], axis=0)
                distances = np.sqrt(np.sum((cluster[:, 1:] - centroid)**2, axis=1))
                sorted_indices = np.argsort(distances)
                num_points_to_keep = int(len(clusters[i]) * alpha)
                clusters[i] = set(np.array(list(clusters[i]))[sorted_indices][:num_points_to_keep])

    # Step 7: Assign final cluster labels
    cluster_labels = np.zeros(len(data), dtype=int)
    for i, c in enumerate(clusters):
        for j in c:
            cluster_labels[j] = i
    return cluster_labels

K = int(arr.shape[0]/100) #here, K is chunks which is data/K1
tr_splits = np.array_split(tr_data, K)
cure(tr_splits[0], 10, 0.5, True)

K = int(arr.shape[0]/200) #here, K is chunks which is data/K1
tr_splits = np.array_split(tr_data, K)
cure(tr_splits[0], 10, 0.5, True)

K = int(arr.shape[0]/500) #here, K is chunks which is data/K1
tr_splits = np.array_split(tr_data, K)
cure(tr_splits[0], 10, 0.5, True)

