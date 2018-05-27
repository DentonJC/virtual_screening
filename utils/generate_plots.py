import os
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from moloi.data_processing import get_data, clean_data
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

datasets = ['bace', 'tox21']#, 'HIV']
# datasets = ['muv', 'clintox']
#descriptors = [['rdkit', 'morgan', 'mordred', 'maccs'], ['rdkit'], ['morgan'], ['mordred'], ['maccs']]
descriptors = [['rdkit', 'morgan', 'mordred', 'maccs']]
splits = ['scaffold', 'cluster', 'random', 'stratified']
for data in datasets:
    path = root_address+"/data/data_configs/"+data+".ini"
    for d in descriptors:
        for s in splits:
            print("Dataset:",str(data))
            print("Descriptors:",str(d))
            logger.info("Dataset: %s",str(data))
            logger.info("Descriptors %s",str(d))
            logger.info("Split: %s",str(s))
            logger.info("Path: %s",str(path))
            logger.info("Split size: %s",str(split_s))
            if not os.path.exists(root_address+"/etc/img/"+data+"/pca"):
                os.makedirs(root_address+"/etc/img/"+data+"/pca")
            if not os.path.exists(root_address+"/etc/img/"+data+"/tsne"):
                os.makedirs(root_address+"/etc/img/"+data+"/tsne")
            x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape, smiles = get_data(logger, path, n_bits, [0], 1337, s, split_s, 10, d, -1)
            
            x_train = clean_data(x_train)
            x_test = clean_data(x_test)
            x_val = clean_data(x_val)
            
            # Scale
            transformer_X = MinMaxScaler().fit(x_train)
            x_train = transformer_X.transform(x_train)
            x_test = transformer_X.transform(x_test)
            x_val = transformer_X.transform(x_val)
            
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

            # plot_TSNE(X.T, Y.T, root_address+"/etc/img/"+data+"/tsne/t-SNE_activity"+str(d)+"_"+s+".png", title="t-SNE_"+data+"_"+str(d)+"_"+s, label_1="active", label_2="inactive", s=1, alpha=0.8)
            # plot_PCA(X.T, Y.T, root_address+"/etc/img/"+data+"/pca/PCA_activity"+str(d)+"_"+s+".png", title="PCA_"+data+"_"+str(d)+"_"+s, label_1="active", label_2="inactive", s=1, alpha=0.8)
            plot_TSNE(X.T, Y.T, root_address+"/etc/img/"+data+"/tsne/t-SNE_activity"+s+".png", title="t-SNE "+data+" "+s+" split", label_1="active", label_2="inactive", s=1, alpha=0.8)
            plot_PCA(X.T, Y.T, root_address+"/etc/img/"+data+"/pca/PCA_activity"+s+".png", title="PCA "+data+" "+s+" split", label_1="active", label_2="inactive", s=1, alpha=0.8)

            y_train = [[0]] * len(y_train)
            y_test = [[1]] * len(y_test)
            y_val = [[2]] * len(y_val)

            y_train, y_test, y_val = np.array(y_train), np.array(y_test), np.array(y_val)
            
            X = np.c_[x_train.T, x_test.T]
            X = np.c_[X, x_val.T]
            Y = np.c_[y_train.T, y_test.T]
            Y = np.c_[Y, y_val.T]

            # plot_TSNE(X.T, Y.T, root_address+"/etc/img/"+data+"/tsne/t-SNE_split"+str(d)+"_"+s+".png", title="t-SNE_"+data+"_"+str(d)+"_"+s, label_1="train", label_2="test", label_3="val", s=1, alpha=0.8)
            # plot_PCA(X.T, Y.T, root_address+"/etc/img/"+data+"/pca/PCA_split"+str(d)+"_"+s+".png", title="PCA_"+data+"_"+str(d)+"_"+s, label_1="train", label_2="test", label_3="val", s=1, alpha=0.8)
            plot_TSNE(X.T, Y.T, root_address+"/etc/img/"+data+"/tsne/t-SNE_split"+s+".png", title="t-SNE "+data+" "+s+" split", label_1="train", label_2="test", label_3="val", s=1, alpha=0.8)
            plot_PCA(X.T, Y.T, root_address+"/etc/img/"+data+"/pca/PCA_split"+s+".png", title="PCA "+data+" "+s+" split", label_1="train", label_2="test", label_3="val", s=1, alpha=0.8)
