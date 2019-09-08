import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy import spatial

def pairwise_dist(manifold_type, x):
    """
    pairwise distance, dim(x) = (n, d)
    :param manifold_type: manifold type, 'Euclidean' or 'Lorentz'
    :param x: input numpy array
    :return: pairwise distance matrix
    """
    if manifold_type == 'Lorentz':
        x0 = x[:,0].reshape(-1,1)
        x1 = x[:,1:]
        m = np.matmul(x1, x1.transpose()) - np.matmul(x0, x0.transpose())
        np.fill_diagonal(m, -1-1e-12)
        m = -m
        dm = np.log(m + np.sqrt(m ** 2 - 1))
        return upper_tri_indexing(dm) # convert to condense form
    elif manifold_type == 'Euclidean':
        dc = spatial.distance.pdist(x, metric='euclidean')
        return dc

def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

def clustering_z(x, z, manifold_type, number_components):
    if manifold_type == 'Euclidean':
        euc_dist_condensed = pairwise_dist(manifold_type, z)
        euc_dist_mtrx = spatial.distance.squareform(euc_dist_condensed)
        cluster = AgglomerativeClustering(number_components, affinity='precomputed', linkage='complete')
        cluster_euc = cluster.fit(euc_dist_mtrx)
        labels = cluster_euc.labels_
        centroid_array = get_centroids(x, z, labels, number_components, manifold_type)
    elif manifold_type == 'Lorentz':
        lor_dist_condensed = pairwise_dist(manifold_type, z)
        lor_dist_mtrx = spatial.distance.squareform(lor_dist_condensed)
        cluster = AgglomerativeClustering(number_components, affinity='precomputed', linkage='complete')
        cluster_lor = cluster.fit(lor_dist_mtrx)
        labels = cluster_lor.labels_
        centroid_array = get_centroids(x, z, labels, number_components, manifold_type)
    return centroid_array

def get_centroids(x, z, labels, number_components, manifold_type):
    centroid_lst = []
    for k in range(number_components):
        index = labels == k
        index = index.reshape(-1,1)

        idx_x = index.repeat(x.shape[1],axis=1)
        idx_z = index.repeat(z.shape[1],axis=1)

        xk = x[idx_x].reshape((-1, x.shape[1]))
        zk = z[idx_z].reshape((-1, z.shape[1]))

        euc_dist_condensed = pairwise_dist(manifold_type, zk)
        euc_dist_mtrx = spatial.distance.squareform(euc_dist_condensed)
        centroid_idx = np.argmin(np.sum(euc_dist_mtrx, axis=0))

        centroid_xk = xk[centroid_idx, :]
        centroid_lst.append(centroid_xk)

    centroid_array = np.concatenate(centroid_lst).reshape(number_components,x.shape[1])
    return centroid_array