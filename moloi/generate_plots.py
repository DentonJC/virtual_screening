import os
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from moloi.data_processing import get_data, clean_data
from moloi.plots import plot_TSNE, plot_PCA


def generate_decomposition(datasets, descriptors, splits, options, random_state, verbose):
    root_address = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
    if not os.path.exists(root_address+"/etc/img/coordinates/"):
        os.makedirs(root_address+"/etc/img/coordinates/")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    if os.path.isfile(root_address + '/etc/img/log'):
        os.remove(root_address + '/etc/img/log')
    handler = logging.FileHandler(root_address + '/etc/img/log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    for dataset in datasets:
        path = root_address+"/data/data_configs/"+dataset+".ini"
        for d in descriptors:
            for s in splits:
                if not os.path.exists(root_address+"/etc/img/"+dataset+"/"+str(d)+"/pca"):
                    os.makedirs(root_address+"/etc/img/"+dataset+"/"+str(d)+"/pca")
                if not os.path.exists(root_address+"/etc/img/"+dataset+"/"+str(d)+"/tsne"):
                    os.makedirs(root_address+"/etc/img/"+dataset+"/"+str(d)+"/tsne")
                if not os.path.exists(root_address+"/etc/img/coordinates/"+dataset+"/"+str(d)+"/pca"):
                    os.makedirs(root_address+"/etc/img/coordinates/"+dataset+"/"+str(d)+"/pca")
                if not os.path.exists(root_address+"/etc/img/coordinates/"+dataset+"/"+str(d)+"/tsne"):
                    os.makedirs(root_address+"/etc/img/coordinates/"+dataset+"/"+str(d)+"/tsne")
                
                data = get_data(logger, options, random_state, verbose)
                y_val, y_train, y_test = data["y_val"], data["y_train"], data["y_test"]
                smiles = data["full_smiles_train"]

                x_train = clean_data(data["x_train"])
                x_test = clean_data(data["x_test"])
                x_val = clean_data(data["x_val"])

                # Scale
                transformer_X = MinMaxScaler().fit(x_train)
                x_train = transformer_X.transform(x_train)
                x_test = transformer_X.transform(x_test)
                x_val = transformer_X.transform(x_val)

                # if s != 'cluster':
                if True:
                    X = np.c_[x_train.T, x_test.T]
                    X = np.c_[X, x_val.T]
                    Y = np.c_[y_train.T, y_test.T]
                    Y = np.c_[Y, y_val.T]

                    y_train = [[0]] * len(y_train)
                    y_test = [[1]] * len(y_test)
                    y_val = [[2]] * len(y_val)

                    y_train, y_test, y_val = np.array(y_train), np.array(y_test), np.array(y_val)

                    Y_a = np.c_[y_train.T, y_test.T]
                    Y_a = np.c_[Y_a, y_val.T]

                    addresses_tsne = root_address+"/etc/img/"+dataset+"/"+str(d)+"/tsne/t-SNE_"+s+".png"
                    titles_tsne = ["t-SNE "+dataset+" "+s+" activity", "t-SNE "+dataset+" "+s+" split"]

                    addresses_pca = root_address+"/etc/img/"+dataset+"/"+str(d)+"/pca/PCA_"+s+".png"
                    titles_pca = ["PCA "+dataset+" "+s+" activity", "PCA "+dataset+" "+s+" split"]
                    
                    labels1 = ["active", "train"]
                    labels2 = ["inactive", "test"]
                    labels3 = [False, "val"]
                    s = 3
                    alpha = 0.5
                    if not os.path.isfile(addresses_tsne):
                        plot_TSNE(X.T, Y.T, Y_a.T, addresses_tsne, titles=titles_tsne, label_1=labels1, label_2=labels2, label_3=labels3, s=s, alpha=alpha)
                    if not os.path.isfile(addresses_pca):
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

