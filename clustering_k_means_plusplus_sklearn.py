import math
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist



## param:
## keepClusterCount: 
##      True, keep the length of clusters, if a cluster has only one node, then select one nearest node from other clusters to merge into this cluster
##      False, if a cluster has only one node, then merge this node into the nearest cluster
## Clusters: like [{'points':[[x,y],[x,y],...],'centroid':[x,y]},{'points':[[x,y],[x,y],...],'centroid':[x,y]},...]
##      Clusters is a list.
##      Each element is a dict, which contains the points of each cluster and the centroid of the cluster. 
def cluster_tuning_for_one_node_cluster(clusters,keepClusterCount=True):
    if keepClusterCount:
        
        while True:
            has_one_node_cluster=False

            for i in range(len(clusters)):
                if len(clusters[i]['points']) == 1:
                    has_one_node_cluster=True
                    distances = []
                    for j in range(len(clusters)):
                        if i == j:
                            continue
                        if len(clusters[j]['points']) > 2:
                            for k in range(len(clusters[j]['points'])):
                                distances.append((j, np.linalg.norm(np.array(clusters[i]['points'][0]) - np.array(clusters[j]['points'][k])),clusters[j]['points'][k]))
                    ## sort distances by the distance ascending
                    distances.sort(key=lambda x: x[1])

                    # merge the nearest node into the cluster
                    if len(distances) > 0:
                        nearest_cluster_idx = distances[0][0]
                        nearest_node = distances[0][2]
                        clusters[i]['points'].append(nearest_node)
                        clusters[nearest_cluster_idx]['points'].remove(nearest_node)

                        ## re-calculate the centroid of the modified two clusters.
                        clusters[i]['centroid'] = np.mean(clusters[i]['points'], axis=0)
                        clusters[nearest_cluster_idx]['centroid'] = np.mean(clusters[nearest_cluster_idx]['points'], axis=0)
            
            if not has_one_node_cluster:
                break
            
    else:
        while True:
            has_one_node_cluster=False
            for i in range(len(clusters)):
                if len(clusters[i]['points']) == 1:
                    has_one_node_cluster=True
                    # find the nearest nodes and its cluster
                    distances = []
                    for j in range(len(clusters)):
                        if i == j:
                            continue
                        for k in range(len(clusters[j]['points'])):
                            distances.append((j, np.linalg.norm(np.array(clusters[i]['points'][0]) - np.array(clusters[j]['points'][k]))))
                    distances.sort(key=lambda x: x[1])

                    if len(distances) > 0:
                        nearest_cluster_idx = distances[0][0]
                        ## merge the single node in current cluster i into the nearest cluster
                        clusters[nearest_cluster_idx]['points'].append(clusters[i]['points'][0])
                        clusters.pop(i)

                        if i<nearest_cluster_idx:
                            nearest_cluster_idx-=1

                        ## recalcuate the centroid of the nearest cluster
                        clusters[nearest_cluster_idx]['centroid'] = np.mean(clusters[nearest_cluster_idx]['points'], axis=0)
                if has_one_node_cluster:
                    break

            if not has_one_node_cluster:
                break
    
    return clusters

    

## param data, 2d numpy array
## return: clusters, list[dict,dict,...], each dict contains two keys: 'points' and 'centroid'
## clusters = [{'points':[[x,y],[x,y],...],'centroid':[x,y]},{'points':[[x,y],[x,y],...],'centroid':[x,y]},...]
def k_means_plusplus_clustering_sklearn(data, k, max_iterations=100, n_jobs=-1):
    # Create a KMeans instance with K-means++ initialization and other specified parameters
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=max_iterations, random_state=42)
    
    # Fit the model to the data
    kmeans.fit(data)
    
    # Extract cluster information
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    # Organize the results into the specified format in parallel
    result = Parallel(n_jobs=n_jobs)(delayed(_get_cluster_info)(data, cluster_labels, cluster_centers, i) for i in range(k))
    
    return result

def _get_cluster_info(data, labels, centers, i):
    return {'points': data[labels == i].tolist(), 'centroid': centers[i].tolist()}
