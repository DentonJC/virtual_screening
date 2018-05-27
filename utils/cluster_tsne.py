"""
Self-contained implementation of the cluster split.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from moloi.data_processing import get_data, clean_data
from sklearn.preprocessing import MinMaxScaler


from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

import numpy as np
import tqdm
from collections import Counter
import subprocess
import tempfile
import os
from scipy.sparse import csr_matrix

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt


def filter_smiles(arr_smiles, arr_uids):
    temp_dir = tempfile.mkdtemp()
    smiles_filename = os.path.join(temp_dir, "smiles.smi")
    output_filename = os.path.join(temp_dir, "output.smi")
    _dump_smi(arr_smiles, smiles_filename, arr_uids)
    subprocess.call(["babel", "-e", smiles_filename, output_filename])
    correct_smiles, correct_uids = _load_smi(output_filename)
    return correct_smiles, correct_uids


def molprint2d_count_fingerprinter(arr_smiles):
    """ Calculate MolPrint2D count fingerprint.
    Parameters
    ----------
    arr_smiles : numpy.array
    Returns
    -------
    result : csr_matrix, numpy.array
    """
    temp_dir = tempfile.mkdtemp()
    smiles_filename = os.path.join(temp_dir, "smiles.smi")
    output_filename = os.path.join(temp_dir, "output.mpd")

    _dump_smi(arr_smiles, smiles_filename, arr_uids=None)
    subprocess.call(["babel", smiles_filename, output_filename])
    fingerprint, columns, _ = _load_mpd(output_filename)

    os.remove(smiles_filename)
    os.remove(output_filename)

    return fingerprint, columns


def _dump_smi(arr_smiles, filename, arr_uids=None):
    assert len(arr_smiles.shape) == 1
    assert (arr_uids is None) or (arr_uids.shape == arr_smiles.shape)
    with open(filename, 'w') as f_out:
        if arr_uids is None:
            arr_uids = np.arange(0, len(arr_smiles)).astype(np.str)
        for uid, smiles in zip(arr_uids, arr_smiles):
            f_out.write(smiles + "\t" + uid + "\n")


def _load_smi(filename):
    uids = []
    l_smiles = []
    with open(filename, 'r') as f_in:
        for line in tqdm.tqdm(f_in):
            line = line.rstrip().split()
            l_smiles.append(line[0])
            uids.append(line[1])
    arr_uids = np.array(uids)
    arr_smiles = np.array(l_smiles)
    return arr_smiles, arr_uids


def _load_mpd(output_filename):
    """
        return csr_matrix (data), numpy.array (columns), numpy.array (rows)
    """

    data = []
    row_idx = []
    col_idx = []

    h_axis = []
    v_axis = []
    _curr_h_axis_idx = 0
    _h_axis_idx = {}

    with open(output_filename, 'r') as f_in:
        for i, line in tqdm.tqdm(enumerate(f_in)):
            line = line.rstrip()
            line = line.split('\t')
            v_axis.append(line[0])
            for key, value in Counter(line[1:]).items():
                row_idx.append(i)
                if not key in _h_axis_idx:
                    _h_axis_idx[key] = _curr_h_axis_idx
                    _curr_h_axis_idx += 1
                col_idx.append(_h_axis_idx[key])
                data.append(value)

    h_axis = np.array([tup[0] for tup in sorted(_h_axis_idx.items(), key=lambda x: x[1])], dtype=np.str)
    v_axis = np.array(v_axis, dtype=np.str)
    arr = csr_matrix((data, (row_idx, col_idx)), shape=(len(v_axis), len(h_axis)))

    return arr, h_axis, v_axis

def _custom_toarray(m):
    m = m.tocoo(copy=False)
    result = np.zeros(m.shape, dtype=m.dtype)
    for idx in range(len(m.data)):
        result[m.row[idx], m.col[idx]] = m.data[idx]
    return result

def tanimoto_similarity(X1T, X2T, fallback_to_custom_toarray=False):
    assert(X1T.dtype in [np.float16, np.float32, np.float64]), \
        "Tanimoto similarity: vectors must be float32 or float64"
    assert(X2T.dtype in [np.float16, np.float32, np.float64]), \
        "Tanimoto similarity: vectors must be float32 or float64"
    # SQUARE ELEMENT-WISE AND SUM
    if sparse.issparse(X1T):
        X1T_copy = X1T.copy()
        X1T_copy.data **= 2
        X1T_sums = np.array(X1T_copy.sum(axis=1))
        del X1T_copy
    else:
        X1T_sums = np.array(np.square(X1T).sum(axis=1))
    if sparse.issparse(X2T):
        X2T_copy = X2T.copy()
        X2T_copy.data **= 2
        X2T_sums = np.array(X2T_copy.sum(axis=1))
        del X2T_copy
    else:
        X2T_sums = np.array(np.square(X2T).sum(axis=1))
    K = X1T.dot(X2T.T)
    if hasattr(K, "toarray"):
        try:
            K = K.toarray()
        except ValueError:
            if fallback_to_custom_toarray is True:
                logger.warning("Fallback to _custom_toarray !")
                K = _custom_toarray(K)
            else:
                raise
    # it seems that true_divide still makes a copy of K
    with np.errstate(divide='ignore'):
        np.true_divide(
            K,
            -K + X1T_sums.reshape(-1, 1) + X2T_sums.reshape(1, -1),
            K)
    # 0/0 == 1 for tanimoto, because all vectors have norm == 1
    K[np.isnan(K)] = 1.0
    return K

def _kernel_to_distance(kernel):
    # returns SQUARED norm
    return np.diagonal(kernel).reshape(-1, 1) + np.diagonal(kernel).reshape(1, -1) - 2 * kernel

def _reduce_kernel_2(kernel, groups):
    u_groups, idx, counts = np.unique(groups, return_inverse=True, return_counts=True)
    reduced_kernel = np.zeros((len(u_groups), len(u_groups)))
    reduced_denom = np.zeros((len(u_groups), len(u_groups)))
    for i in range(len(idx)):
        for j in range(len(idx)):
            reduced_kernel[idx[i], idx[j]] += kernel[i, j]
            reduced_denom[idx[i], idx[j]] += 1
    return np.divide(reduced_kernel, reduced_denom), u_groups, counts

class BalancedAgglomerativeClustering(BaseEstimator):

    class _Tree(object):
        def __init__(self, children, leaf_weights):
            self._INTNAN = -1 # be careful! -1 is a proper index
            self.children = np.array(children)[:]
            self.n_samples = len(leaf_weights)
            self.n_nodes = 2 * self.n_samples - 1
            self.parents = np.empty(self.n_nodes, dtype=np.int64)
            self.parents[:] = self._INTNAN
            for i in range(self.children.shape[0]):
                for j in [0, 1]:
                    self.parents[self.children[i, j]] = i + self.n_samples
            self.leaf_weights = np.array(leaf_weights)[:]
        def get_clusters(self, n_clusters):
            leaf_weights = self.leaf_weights[:]
            clusters = []
            used = set()
            while n_clusters > 0:
                leafs = self.find_cluster(float(leaf_weights.sum()) / float(n_clusters), leaf_weights)
                cluster = []
                for leaf in leafs:
                    if not leaf in used:
                        used.add(leaf)
                        cluster.append(leaf)
                        leaf_weights[leaf] = 0
                clusters.append(cluster)
                n_clusters -= 1
            return clusters
        def find_cluster(self, cluster_size, leaf_weights):
            sizes = self.calculate_cluster_sizes(leaf_weights)
            node = np.abs(sizes - cluster_size).argmin()
            # if some leafs have zero weight , choose highest possible node
            # if node has no siblings, go up
            while self.parents[node] != self._INTNAN and sizes[self.parents[node]] == sizes[node]:
                node = self.parents[node]
            return sorted(self.get_leafs(node))
        def calculate_cluster_sizes(self, leaf_weights):
            sizes = np.zeros(self.n_nodes)
            for i in range(self.n_samples):
                w = leaf_weights[i]
                current = i
                sizes[current] += w
                while self.parents[current] != self._INTNAN:
                    current = self.parents[current]
                    sizes[current] += w
            return sizes
        def get_leafs(self, node):
            leafs = []
            stack = []
            visited = set()
            stack.append(node)
            while len(stack) > 0:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                if current < self.n_samples:
                    leafs.append(current)
                else:
                    stack.append(self.children[current - self.n_samples, 0])
                    stack.append(self.children[current - self.n_samples, 1])
            return leafs

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        assert not (np.diagonal(X) == 0).all(), "'X' must be kernel matrix, not distance!"
        kernel = X
        groups = y
        linkage = "average"
        if groups is None:
            groups = range(kernel.shape[0])
        reduced_kernel, u_groups, counts = _reduce_kernel_2(kernel, groups)
        dist = _kernel_to_distance(reduced_kernel)
        ac = AgglomerativeClustering(affinity="precomputed", linkage=linkage).fit(dist)
        tree = BalancedAgglomerativeClustering._Tree(ac.children_, counts)
        clusters = tree.get_clusters(self.n_clusters)
        new_groups = np.array(groups)
        for i, cluster in enumerate(clusters):
            for ind in cluster:
                new_groups[groups==u_groups[ind]] = i
        self.labels_ = new_groups
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

def cluster_split(smi, test_cluster_id, n_splits):
    X_molprint, _ = molprint2d_count_fingerprinter(np.array(smi))
    X_molprint = X_molprint.astype(np.float32)
    K_molprint_tanimoto = tanimoto_similarity(X_molprint, X_molprint)

    clustering = BalancedAgglomerativeClustering(n_clusters=n_splits)
    logger.info("Clustering")
    labels = clustering.fit_predict(K_molprint_tanimoto)

    train_ids = np.where(np.array(labels) != test_cluster_id)[0]
    test_ids = np.where(np.array(labels) == test_cluster_id)[0]

    return train_ids, test_ids

if __name__ == "__main__":
    # Split
    n_splits = 3

    root_address = os.path.dirname(os.path.realpath(__file__)).replace("/utils", "")
    data_config = "/data/data_configs/bace.ini"
    model_config = "/data/model_configs/configs.ini"
    section = 'SVC'
    descriptors = ['mordred', 'maccs']
    n_bits = 256
    n_cv = 5
    n_iter = 5
    n_jobs = -1
    patience = 100
    metric = 'roc_auc'
    split_type = 'scaffold'
    split_s = 0.1
    targets = [0]
    random_state = 1337
    verbose = 10
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape, smi = get_data(logger, root_address+data_config, n_bits,
                                                                                                 targets, random_state, split_type, split_s,
                                                                                                 verbose, descriptors, n_jobs)

    x_train = clean_data(x_train)
    x_test = clean_data(x_test)
    x_val = clean_data(x_val)
        
    # Scale
    transformer_X = MinMaxScaler().fit(x_train)
    x_train = transformer_X.transform(x_train)
    x_test = transformer_X.transform(x_test)
    x_val = transformer_X.transform(x_val)

    X = x_train
    y = y_train
    
    X_molprint, _ = molprint2d_count_fingerprinter(np.array(smi))
    X_molprint = X_molprint.astype(np.float32)
    K_molprint_tanimoto = tanimoto_similarity(X_molprint, X_molprint)
    print(K_molprint_tanimoto[0])

    clustering = BalancedAgglomerativeClustering(n_clusters=n_splits)

    labels = clustering.fit_predict(K_molprint_tanimoto)

    print(labels)
    
    s=2
    alpha=1
    title_tsne = "t-SNE BACE cluster split"
    title_pca = "PCA BACE cluster split"
    # Plot tSNE
    model = TSNE(metric="precomputed", verbose=10, learning_rate=500)
    X_2d = model.fit_transform(_kernel_to_distance(K_molprint_tanimoto))
    print(X_2d[0])
    plt.scatter(X_2d[np.where(labels==0)[0], 0], X_2d[np.where(labels==0)[0], 1], color="r", label="train", s=s, alpha=alpha)
    plt.scatter(X_2d[np.where(labels==1)[0], 0], X_2d[np.where(labels==1)[0], 1], color="g", label="test", s=s, alpha=alpha)
    plt.scatter(X_2d[np.where(labels==2)[0], 0], X_2d[np.where(labels==2)[0], 1], color="b", label="val", s=s, alpha=alpha)
    plt.title(title_tsne)
    plt.legend()
    plt.savefig("cluster_split_tsne.png", dpi=500)
    plt.clf()
    plt.cla()
    plt.close()
    
    model = PCA()
    X_2d = model.fit_transform(_kernel_to_distance(K_molprint_tanimoto))
    print(X_2d[0])
    plt.scatter(X_2d[np.where(labels==0)[0], 0], X_2d[np.where(labels==0)[0], 1], color="r", label="train", s=s, alpha=alpha)
    plt.scatter(X_2d[np.where(labels==1)[0], 0], X_2d[np.where(labels==1)[0], 1], color="g", label="test", s=s, alpha=alpha)
    plt.scatter(X_2d[np.where(labels==2)[0], 0], X_2d[np.where(labels==2)[0], 1], color="b", label="val", s=s, alpha=alpha)
    plt.title(title_pca)
    plt.legend()
    plt.savefig("cluster_split_pca.png", dpi=500)
    plt.clf()
    plt.cla()
    plt.close()
