import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def group_cluster(labels, data):             # Prepares dictionary of points to labels
  idx = np.arange(data.shape[0])
  clusters = dict()
  for label, id in zip(labels, idx):
    if label in clusters:
      clusters[label].append(id)
    else:
      clusters[label] = [id]
  return clusters

def process_cluster(clusters):               # Segregates the Single Clusters and Multi-Point Clusters
  single_point_clusters, multi_point_clusters = [], []
  for label, point_indices in clusters.items():
    if len(point_indices) == 1:
      single_point_clusters.append(point_indices[0])
    else:
      multi_point_clusters.append(tuple(point_indices))
  return single_point_clusters, multi_point_clusters

def summary(data, point_indices):
  return [len(point_indices), data[point_indices, :].sum(axis=0), np.square(data[point_indices, :]).sum(axis=0)]

def summarise_info_clusters(labels, data):    # Issues the Information corresponding to label-wise points
  compressed = []
  clusters = group_cluster(labels, data)
  for label, point_indices in clusters.items():
    summary_label = summary(data, point_indices)
    compressed.append({'N': summary_label[0],'SUM': summary_label[1],'SUMSQ': summary_label[2]})
  return compressed

def update_cluster(cluster, new_point=None, new_cluster=None):    # Update the cluster's info in terms of either point or cluster
  if new_point is not None and new_cluster is None:
    cluster['N'] += 1
    cluster['SUM'] += new_point
    cluster['SUMSQ'] += np.square(new_point)
    return cluster
  if new_cluster is not None and new_point is None:
    cluster['N'] += new_cluster['N']
    cluster['SUM'] += new_cluster['SUM']
    cluster['SUMSQ'] += new_cluster['SUMSQ']
    return cluster

epsilon = 1e-10

def count_points(cluster_dicts):                                    # Count number of points from different cluster sets (DS & CS)
    return sum([clusters['N'] for clusters in cluster_dicts])

def calc_norm_mahalnobis(point, point_ref, std_ref):                # Returns Mahalnobis Distance 
  return np.sqrt(np.sum(np.square((point - point_ref) / (std_ref+epsilon))))

def calc_mahalnobis_cluster(cluster, point=None, cluster2=None):    # Calculate Mahalnobis Distance from reference cluster to either point or cluster 
  if point is not None and cluster2 is None:
    point_ref = cluster['SUM'] / cluster['N']
    std_ref = np.sqrt((cluster['SUMSQ'] / cluster['N']) - np.square(point_ref))
    return calc_norm_mahalnobis(point, point_ref, std_ref)
  if cluster2 is not None and point is None:
    c1 = cluster['SUM'] / cluster['N']
    c2 = cluster2['SUM'] / cluster2['N']
    std1 = np.sqrt((cluster['SUMSQ'] / cluster['N']) - np.square(c1))
    std2 = np.sqrt((cluster2['SUMSQ'] / cluster2['N']) - np.square(c2))
    return calc_norm_mahalnobis(c1, c2, std2 * std1)

def generate(ds_clusters, cs_clusters, rs_points):
  return {'DS Clusters': len(ds_clusters),
        'DS Points': count_points(ds_clusters),
        'CS Clusters': len(cs_clusters),
        'CS Points': count_points(cs_clusters),
        'RS Points': len(rs_points)}

def statistics(ds_clusters, cs_clusters, rs_points):
  stats = generate(ds_clusters, cs_clusters, rs_points)
  stats['SUM'] = stats['DS Points'] + stats['CS Points'] + stats['RS Points']
  print(stats)
  return stats

def predict(chunks, clusters, data, threshold):
  final_labels = []
  for chunk in range(chunks):
    for point_ind in range(len(data[chunk])):
      min_dist, assign_status = None, None
      for cluster_ind in range(len(clusters)):
        dist = calc_mahalnobis_cluster(clusters[cluster_ind], data[chunk][point_ind, 1:], None)
        if min_dist is None:
          min_dist, assign_status = dist, cluster_ind
          continue
        if dist < min_dist:
          min_dist, assign_status = dist, cluster_ind
      if min_dist < threshold:
        final_labels.append((int(data[chunk][point_ind, 0]), assign_status))
      else:
        final_labels.append((int(data[chunk][point_ind, 0]), -1))
  final_labels = sorted(final_labels, key=lambda x: x[0])
  return final_labels

def KM(data, num_clusters):
  df = data[:, 1:]
  kmeans = KMeans(n_clusters=num_clusters)
  label = kmeans.fit_predict(df)
  centroids = kmeans.cluster_centers_
  u_labels = np.unique(label)
  for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
  plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
  plt.legend()
  return kmeans

def add_points_to_clusters(points, clusters):
  features_dim = points.shape[1]
  threshold = 2 * np.sqrt(features_dim)
  unassigned_points = list()
  for point_ind in range(len(points)):
    min_dist, assign_status = None, None
    for cluster_ind in range(len(clusters)):
      dist = calc_mahalnobis_cluster(clusters[cluster_ind], points[point_ind], None)
      if min_dist is None:
        min_dist, assign_status = dist, cluster_ind
        continue
      if dist < min_dist:
        min_dist, assign_status = dist, cluster_ind       
    if min_dist < threshold:
      clusters[assign_status] = update_cluster(clusters[assign_status], points[point_ind], None)
    else:
      unassigned_points.append(point_ind)
  return unassigned_points

def merge_clusters(old_clusters, new_clusters, threshold, return_two):
  merge_tuples = list()
  for new_cluster_ind in range(len(new_clusters)):
    min_dist, assign_status = None, None
    for old_cluster_ind in range(len(old_clusters)):
      dist = calc_mahalnobis_cluster(old_clusters[old_cluster_ind], None, new_clusters[new_cluster_ind])
      if min_dist is None:
        min_dist, assign_status = dist, old_cluster_ind
        continue
      if dist < min_dist:
        min_dist, assign_status = dist, old_cluster_ind
    if min_dist < threshold:
      merge_tuples.append((assign_status, new_cluster_ind))
    else:
      continue
  for old_cluster_ind, new_cluster_ind in merge_tuples:
    old_cluster = old_clusters[old_cluster_ind]
    new_cluster = new_clusters[new_cluster_ind]
    old_clusters[old_cluster_ind] = update_cluster(old_cluster, None, new_cluster)
  for old_cluster_ind, new_cluster_ind in sorted(merge_tuples, key=lambda x: x[1], reverse=True):
    new_clusters.pop(new_cluster_ind)
  if return_two==True:
    return old_clusters, new_clusters
  else:
    old_clusters.extend(new_clusters)
    return old_clusters

cs = pd.read_csv('/content/drive/MyDrive/MLBD_Assignment_1/train.csv')
arr = cs.to_numpy()
tr_label = arr[:,0]
num_clusters = 10

tr_features = PCA(n_components=2).fit_transform(arr[:,1:])
tr_data = np.hstack((tr_label.reshape((tr_label.shape[0],1)), tr_features))

np.random.shuffle(tr_data)

K = int(arr.shape[0]/100) #here, K is chunks which is data/K1
tr_splits = np.array_split(tr_data, K)
threshold = 2 * np.sqrt(tr_splits[0].shape[1] - 1)
kmeans_1st_split = KM(tr_splits[0], num_clusters)

stats, cs_clusters, new_cs_clusters, rs_points = [], [], [], None
ds_clusters = summarise_info_clusters(kmeans_1st_split.labels_, tr_splits[0][:, 1:])
for chunk in range(1, K):
  unassigned_points = add_points_to_clusters(tr_splits[chunk][:, 1:], ds_clusters)
  if len(cs_clusters) > 0:
    if len(unassigned_points):
      unassigned_points = add_points_to_clusters(tr_splits[chunk][unassigned_points, 1:], cs_clusters)
  if len(unassigned_points) > 0:
    if chunk == 1:
      rs_points = tr_splits[chunk][unassigned_points, 1:]
    else:
      rs_points = np.append(rs_points, tr_splits[chunk][unassigned_points, 1:], axis=0)
  if len(rs_points) >= K:
    km = KMeans(K).fit(rs_points)
    rs_idx, cs_idx = process_cluster(group_cluster(km.labels_, rs_points))
    if len(cs_idx) > 0:
      if len(cs_clusters) > 0:
        new_cs_clusters = summarise_info_clusters(km.labels_[tuple(cs_idx)], rs_points[tuple(cs_idx)])
      else:
        cs_clusters = summarise_info_clusters(km.labels_[tuple(cs_idx)], rs_points[tuple(cs_idx)])
      if len(new_cs_clusters):
        cs_clusters = merge_clusters(cs_clusters, new_cs_clusters, threshold, return_two=False)
    if len(rs_idx) > 0:
      rs_points = rs_points[rs_idx]
  if chunk < (K - 1):
    stats.append(statistics(ds_clusters, cs_clusters, rs_points))

ds_clusters, cs_clusters = merge_clusters(ds_clusters, cs_clusters, threshold, return_two=True)
stats.append(statistics(ds_clusters, cs_clusters, rs_points))
predictions = predict(K, ds_clusters, tr_splits, threshold)

K = int(arr.shape[0]/200) #here, K is chunks which is data/K1
tr_splits = np.array_split(tr_data, K)
threshold = 2 * np.sqrt(tr_splits[0].shape[1] - 1)
kmeans_1st_split = KM(tr_splits[0], num_clusters)

stats, cs_clusters, new_cs_clusters, rs_points = [], [], [], None
ds_clusters = summarise_info_clusters(kmeans_1st_split.labels_, tr_splits[0][:, 1:])
for chunk in range(1, K):
  unassigned_points = add_points_to_clusters(tr_splits[chunk][:, 1:], ds_clusters)
  if len(cs_clusters) > 0:
    if len(unassigned_points):
      unassigned_points = add_points_to_clusters(tr_splits[chunk][unassigned_points, 1:], cs_clusters)
  if len(unassigned_points) > 0:
    if chunk == 1:
      rs_points = tr_splits[chunk][unassigned_points, 1:]
    else:
      rs_points = np.append(rs_points, tr_splits[chunk][unassigned_points, 1:], axis=0)
  if len(rs_points) >= K:
    km = KMeans(K).fit(rs_points)
    rs_idx, cs_idx = process_cluster(group_cluster(km.labels_, rs_points))
    if len(cs_idx) > 0:
      if len(cs_clusters) > 0:
        new_cs_clusters = summarise_info_clusters(km.labels_[tuple(cs_idx)], rs_points[tuple(cs_idx)])
      else:
        cs_clusters = summarise_info_clusters(km.labels_[tuple(cs_idx)], rs_points[tuple(cs_idx)])
      if len(new_cs_clusters):
        cs_clusters = merge_clusters(cs_clusters, new_cs_clusters, threshold, return_two=False)
    if len(rs_idx) > 0:
      rs_points = rs_points[rs_idx]
  if chunk < (K - 1):
    stats.append(statistics(ds_clusters, cs_clusters, rs_points))

ds_clusters, cs_clusters = merge_clusters(ds_clusters, cs_clusters, threshold, return_two=True)
stats.append(statistics(ds_clusters, cs_clusters, rs_points))
predictions = predict(K, ds_clusters, tr_splits, threshold)

K = int(arr.shape[0]/500) #here, K is chunks which is data/K1
tr_splits = np.array_split(tr_data, K)
threshold = 2 * np.sqrt(tr_splits[0].shape[1] - 1)
kmeans_1st_split = KM(tr_splits[0], num_clusters)

stats, cs_clusters, new_cs_clusters, rs_points = [], [], [], None
ds_clusters = summarise_info_clusters(kmeans_1st_split.labels_, tr_splits[0][:, 1:])
for chunk in range(1, K):
  unassigned_points = add_points_to_clusters(tr_splits[chunk][:, 1:], ds_clusters)
  if len(cs_clusters) > 0:
    if len(unassigned_points):
      unassigned_points = add_points_to_clusters(tr_splits[chunk][unassigned_points, 1:], cs_clusters)
  if len(unassigned_points) > 0:
    if chunk == 1:
      rs_points = tr_splits[chunk][unassigned_points, 1:]
    else:
      rs_points = np.append(rs_points, tr_splits[chunk][unassigned_points, 1:], axis=0)
  if len(rs_points) >= K:
    km = KMeans(K).fit(rs_points)
    rs_idx, cs_idx = process_cluster(group_cluster(km.labels_, rs_points))
    if len(cs_idx) > 0:
      if len(cs_clusters) > 0:
        new_cs_clusters = summarise_info_clusters(km.labels_[tuple(cs_idx)], rs_points[tuple(cs_idx)])
      else:
        cs_clusters = summarise_info_clusters(km.labels_[tuple(cs_idx)], rs_points[tuple(cs_idx)])
      if len(new_cs_clusters):
        cs_clusters = merge_clusters(cs_clusters, new_cs_clusters, threshold, return_two=False)
    if len(rs_idx) > 0:
      rs_points = rs_points[rs_idx]
  if chunk < (K - 1):
    stats.append(statistics(ds_clusters, cs_clusters, rs_points))

ds_clusters, cs_clusters = merge_clusters(ds_clusters, cs_clusters, threshold, return_two=True)
stats.append(statistics(ds_clusters, cs_clusters, rs_points))
predictions = predict(K, ds_clusters, tr_splits, threshold)
