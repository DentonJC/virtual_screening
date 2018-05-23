import os
import logging
import numpy as np
from moloi.data_processing import get_data
from moloi.plots import plot_TSNE, plot_PCA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

root_address = os.path.dirname(os.path.realpath(__file__)).replace("/utils", "")
n_bits = 32
split_type = 'scaffold'
split_s = 0.1

datasets = ['bace', 'tox21']#, 'HIV']
descriptors = [['rdkit', 'morgan', 'mordred', 'maccs'], ['rdkit'], ['morgan'], ['mordred'], ['maccs']]
splits = ['scaffold', 'cluster', 'random', 'stratified']
for data in datasets:
    path = root_address+"/data/data_configs/"+data+".ini"
    for d in descriptors:
        for s in splits:
            print("Dataset:",data)
            print("Descriptors",str(d))
            print("Split:",s)            
            x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape, smiles = get_data(logger, path, n_bits, [0], 1337, s, split_s, 10, d, -1)
            
            X = np.c_[x_train.T, x_test.T]
            X = np.c_[X, x_val.T]
            Y = np.c_[y_train.T, y_test.T]
            Y = np.c_[Y, y_val.T]

            plot_TSNE(X.T, Y.T, root_address+"/etc/img/t-SNE_activity_"+data+"_"+str(d)+"_"+s+".png", "t-SNE_"+data+"_"+str(d)+"_"+s)
            plot_PCA(X.T, Y.T, root_address+"/etc/img/PCA_activity_"+data+"_"+str(d)+"_"+s+".png", "PCA_"+data+"_"+str(d)+"_"+s)

            y_train = [[0]] * len(y_train)
            y_test = [[1]] * len(y_test)
            
            y_train, y_test = np.array(y_train), np.array(y_test)
            
            X = np.c_[x_train.T, x_test.T]
            Y = np.c_[y_train.T, y_test.T]

            plot_TSNE(X.T, Y.T, root_address+"/etc/img/t-SNE_split_"+data+"_"+str(d)+"_"+s+".png", "t-SNE_"+data+"_"+str(d)+"_"+s, "train", "test")
            plot_PCA(X.T, Y.T, root_address+"/etc/img/PCA_split_"+data+"_"+str(d)+"_"+s+".png", "PCA_"+data+"_"+str(d)+"_"+s, "train", "test")
