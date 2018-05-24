import os
import logging
import numpy as np
from moloi.data_processing import get_data, preprocessing
from moloi.plots import plot_TSNE, plot_PCA

root_address = os.path.dirname(os.path.realpath(__file__)).replace("/utils", "")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

if os.path.isfile(root_address + '/etc/img/log'):
    os.remove(root_address + '/etc/img/log')
handler = logging.FileHandler(root_address + '/etc/img/log')
handler.setFormatter(formatter)
logger.addHandler(handler)

n_bits = 32
split_type = 'scaffold'
split_s = 0.1

# datasets = ['bace', 'tox21']#, 'HIV']
datasets = ['muv', 'clintox']
descriptors = [['rdkit', 'morgan', 'mordred', 'maccs'], ['rdkit'], ['morgan'], ['mordred'], ['maccs']]
splits = ['scaffold', 'cluster', 'random', 'stratified']
for data in datasets:
    path = root_address+"/data/data_configs/"+data+".ini"
    for d in descriptors:
        for s in splits:
            print("Dataset: %s",str(data))
            print("Descriptors %s",str(d))
            logger.info("Dataset: %s",str(data))
            logger.info("Descriptors %s",str(d))
            logger.info("Split: %s",str(s))
            logger.info("Path: %s",str(path))
            logger.info("Split size: %s",str(split_s))
            x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape, smiles = get_data(logger, path, n_bits, [0], 1337, s, split_s, 10, d, -1)
            
            x_train = preprocessing(x_train)
            x_test = preprocessing(x_test)
            x_val = preprocessing(x_val)
            
            X = np.c_[x_train.T, x_test.T]
            X = np.c_[X, x_val.T]
            Y = np.c_[y_train.T, y_test.T]
            Y = np.c_[Y, y_val.T]
            
            logger.info("Shapes in generate_plots script: %s",str(data))
            logger.info("SMILES shape: %s", str(smiles.shape))
            logger.info("x_train shape: %s", str(x_train.shape))
            logger.info("x_test shape: %s", str(x_test.shape))
            logger.info("x_val hape: %s", str(x_val.shape))
            logger.info("y_train shape: %s", str(y_train.shape))
            logger.info("y_test shape: %s", str(y_test.shape))
            logger.info("y_val shape: %s", str(y_val.shape))
            logger.info("X shape: %s", str(X.shape))
            logger.info("Y shape: %s", str(Y.shape)+'\n')

            plot_TSNE(X.T, Y.T, root_address+"/etc/img/t-SNE_activity_"+data+"_"+str(d)+"_"+s+".png", "t-SNE_"+data+"_"+str(d)+"_"+s)
            plot_PCA(X.T, Y.T, root_address+"/etc/img/PCA_activity_"+data+"_"+str(d)+"_"+s+".png", "PCA_"+data+"_"+str(d)+"_"+s)

            y_train = [[0]] * len(y_train)
            y_test = [[1]] * len(y_test)
            
            y_train, y_test = np.array(y_train), np.array(y_test)
            
            X = np.c_[x_train.T, x_test.T]
            Y = np.c_[y_train.T, y_test.T]

            plot_TSNE(X.T, Y.T, root_address+"/etc/img/t-SNE_split_"+data+"_"+str(d)+"_"+s+".png", "t-SNE_"+data+"_"+str(d)+"_"+s, "train", "test")
            plot_PCA(X.T, Y.T, root_address+"/etc/img/PCA_split_"+data+"_"+str(d)+"_"+s+".png", "PCA_"+data+"_"+str(d)+"_"+s, "train", "test")
