
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

## data, 2d numpy array
## return: clusters, list[dict,dict,...], each dict contains two keys: 'points' and 'centroid'
## clusters = [{'points':[[x,y],[x,y],...],'centroid':[x,y]},{'points':[[x,y],[x,y],...],'centroid':[x,y]},...]
def k_means_clustering_sklearn(data, k, max_iterations=100, n_jobs=-1):
    # Create a KMeans instance with random initialization and other specified parameters
    kmeans = KMeans(n_clusters=k, init='random', max_iter=max_iterations, random_state=42)
    
    # Fit the model to the data
    kmeans.fit(data)
    
    # Extract cluster information
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    # Organize the results into the specified format in parallel
    clusters = Parallel(n_jobs=n_jobs)(delayed(_get_cluster_info)(data, cluster_labels, cluster_centers, i) for i in range(k))
    

    return clusters

def _get_cluster_info(data, labels, centers, i):
    return {'points': data[labels == i].tolist(), 'centroid': centers[i].tolist()}



