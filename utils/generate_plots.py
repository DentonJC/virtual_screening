import os
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from moloi.data_processing import get_data, clean_data
from moloi.plots import plot_TSNE, plot_PCA
from moloi.splits.cluster_split import cluster_split, molprint2d_count_fingerprinter, BalancedAgglomerativeClustering, tanimoto_similarity, _kernel_to_distance

root_address = os.path.dirname(os.path.realpath(__file__)).replace("/utils", "")
if not os.path.exists(root_address+"/etc/img/"):
    os.makedirs(root_address+"/etc/img/")

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

datasets = ['clintox', 'bace']
descriptors = [['rdkit', 'morgan', 'mordred', 'maccs'], ['rdkit'], ['morgan'], ['mordred'], ['maccs']]
#descriptors = [['rdkit', 'morgan', 'mordred', 'maccs']]
splits = ['cluster', 'scaffold', 'random', 'stratified']
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
            if not os.path.exists(root_address+"/etc/img/"+data+"/"+str(d)+"/pca"):
                os.makedirs(root_address+"/etc/img/"+data+"/"+str(d)+"/pca")
            if not os.path.exists(root_address+"/etc/img/"+data+"/"+str(d)+"/tsne"):
                os.makedirs(root_address+"/etc/img/"+data+"/"+str(d)+"/tsne")
            x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape, smiles = get_data(logger, path, n_bits, [0], 1337, s, split_s, 10, d, -1)
            
            x_train = clean_data(x_train)
            x_test = clean_data(x_test)
            x_val = clean_data(x_val)
            
            # Scale
            transformer_X = MinMaxScaler().fit(x_train)
            x_train = transformer_X.transform(x_train)
            x_test = transformer_X.transform(x_test)
            x_val = transformer_X.transform(x_val)
            
            #if s != 'cluster':
            if True:
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
                
                y_train = [[0]] * len(y_train)
                y_test = [[1]] * len(y_test)
                y_val = [[2]] * len(y_val)
                
                y_train, y_test, y_val = np.array(y_train), np.array(y_test), np.array(y_val)
                
                Y_a = np.c_[y_train.T, y_test.T]
                Y_a = np.c_[Y_a, y_val.T]
                
                logger.info("y_train shape: %s", str(y_train.shape))
                logger.info("y_test shape: %s", str(y_test.shape))
                logger.info("y_val shape: %s", str(y_val.shape))
                logger.info("X shape: %s", str(X.shape))
                logger.info("Y shape: %s", str(Y.shape)+'\n')
                
                addresses_tsne = [root_address+"/etc/img/"+data+"/"+str(d)+"/tsne/t-SNE_activity_"+s+".png", root_address+"/etc/img/"+data+"/"+str(d)+"/tsne/t-SNE_split_"+s+".png"]
                titles_tsne = ["t-SNE "+data+" "+s+" split", "t-SNE "+data+" "+s+" activity"]
                
                addresses_pca = [root_address+"/etc/img/"+data+"/"+str(d)+"/pca/PCA_activity_"+s+".png", root_address+"/etc/img/"+data+"/"+str(d)+"/pca/PCA_split_"+s+".png"]
                titles_pca = ["PCA "+data+" "+s+" split", "PCA "+data+" "+s+" activity"]
                
                labels1 = ["active", "train"]
                labels2 = ["inactive", "test"]
                labels3 = [False, "val"]
                s = 3
                alpha = 0.5
                
                plot_TSNE(X.T, Y.T, Y_a.T, addresses_tsne, titles=titles_tsne, label_1=labels1, label_2=labels2, label_3=labels3, s=s, alpha=alpha)
                plot_PCA(X.T, Y.T, Y_a.T, addresses_pca, titles=titles_pca, label_1=labels1, label_2=labels2, label_3=labels3, s=s, alpha=alpha)
            
            # else:
            #     n_splits = 3
            #    
            #     X_molprint, _ = molprint2d_count_fingerprinter(np.array(smiles))
            #     X_molprint = X_molprint.astype(np.float32)
            #     K_molprint_tanimoto = tanimoto_similarity(X_molprint, X_molprint)
            #     clustering = BalancedAgglomerativeClustering(n_clusters=n_splits)
            #     Y = clustering.fit_predict(K_molprint_tanimoto)
            #     X = _kernel_to_distance(K_molprint_tanimoto)
            #     
            #     plot_TSNE(X, Y, root_address+"/etc/img/"+data+"/"+str(d)+"/tsne/t-SNE_split_"+s+".png", title="t-SNE "+data+" "+s+" split", label_1="train", label_2="test", label_3="val", s=1, alpha=0.8)
            #     plot_PCA(X, Y, root_address+"/etc/img/"+data+"/"+str(d)+"/pca/PCA_split_"+s+".png", title="PCA "+data+" "+s+" split", label_1="train", label_2="test", label_3="val", s=1, alpha=0.8)
            
