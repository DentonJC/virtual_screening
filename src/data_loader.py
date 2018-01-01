#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src._desc_rdkit import smiles_to_desc_rdkit
from src.main import drop_nan
from sklearn import preprocessing


Dummy_n = 3000
n_physical = 196

def featurization(logger, filename, DUMMY, fingerprint, nBits, path):
    data = pd.read_csv(filename)
    smiles = []
    if DUMMY:
        data = data[:Dummy_n]

    logger.info("Loading data")
    l_headers = list(data)
    if " 'f'" in l_headers or " 'p'" in l_headers:
        data = np.array(data)
        _, cols = data.shape
        features = np.empty(shape=(data.shape[0], ))
        labels = np.empty(shape=(data.shape[0], ))
        for i in range(cols):
            if " 'f'" in l_headers[i] or " 'p'" in l_headers[i]:
                features = np.c_[features,data[:,i]]
            else:
                labels = np.c_[labels,data[:,i]]
        features = np.delete(features, [0], axis=1)
        labels = np.delete(labels, [0], axis=1)
        
    else:
        logger.info("Physic data extraction")
        smiles = data["smiles"]
        data = data.drop("smiles", 1)
        if "mol_id" in list(data):
            data = data.drop("mol_id", 1)
        l_headers = list(data)
        physic_smiles = pd.Series(smiles)
        physic_data, missing = smiles_to_desc_rdkit(physic_smiles)
        smiles = np.array(smiles)
        smiles = np.delete(smiles, missing)
        data = np.array(data)
        data = np.delete(data, missing, axis=1)
        p_headers = "p" * physic_data.shape[1]
        _, cols = data.shape
        l = []
        for i in range(cols):
            if isinstance(data[i][0], str):
                le = LabelEncoder()
                le.fit(data[:,i])
                c = le.transform(data[:,i])
                l.append(c)
            else:
                l.append(data[:, i])
            
        labels = np.array(l).T
        labels = np.delete(labels, missing, axis=0)

        logger.info("Featurization")
        ms = [Chem.MolFromSmiles(x) for x in smiles]
        if fingerprint in ['MACCS', 'maccs', 'Maccs', 'maccs (167)']:
            features = [MACCSkeys.GenMACCSKeys(x) for x in ms if x]
            features = np.array(features)
            f_headers = "f" * features.shape[1]
            features = np.c_[features, physic_data]
            featurized = np.c_[features, labels]
            filename = filename.replace(".csv", "_maccs.csv")

        if fingerprint in ['MORGAN', 'Morgan', 'morgan', 'morgan (n)']:
            features = [AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=nBits) for x in ms if x]
            features = np.array(features)
            f_headers = "f" * features.shape[1]
            features = np.c_[features, physic_data]
            featurized = np.c_[features, labels]
            filename = filename.replace(".csv", "_morgan_"+str(nBits)+".csv")
        
        head, _sep, tail = filename.rpartition('/')
        filename = path + "/data/preprocessed/" + tail
        headers = list(f_headers) + list(p_headers) + list(l_headers)            
        headers = str(headers).replace('[','').replace(']','').replace('#','')
        np.savetxt(filename, featurized, delimiter=",", header=headers, fmt='%3f')

        if DUMMY:
            features = np.array(features[:Dummy_n])
            labels = np.array(labels[:Dummy_n])
        
    return data, features, labels
        

def preprocessing(logger, features, nBits):
    st = StandardScaler()
    remove_rows = []
    
    features = features.T
    
    for i in range(nBits, features.shape[0]):
        try:
            st.fit_transform(features[i].reshape(1, -1)) # it's do nothing, really
            #features[i] = preprocessing.normalize(features[i].reshape(1, -1))
        except:
            remove_rows.append(i)
            logger.info("Input contains NaN, infinity or a value too large for dtype('float64')")
    features = features.T
    features = np.delete(features, remove_rows, axis=1)
    #np.savetxt("out.csv", features, delimiter=",", fmt='%3f')
    
    
def processing(logger, filename, nBits, set_targets, set_features, DUMMY, fingerprint, path):
    data, features, labels = featurization(logger, filename, DUMMY, fingerprint, nBits, path)
    
    logger.info("Data shape: %s", str(data.shape))
    logger.info("Features shape: %s", str(features.shape))
    logger.info("Labels shape: %s", str(labels.shape))
    logger.info("Data loaded")
    
    preprocessing(logger, features, nBits)
    logger.info("Data preprocessed")

    if set_targets:
        labels = labels[:, set_targets].reshape(labels.shape[0], len(set_targets))
    if set_features in ['physical', 'p']:
        features = features[:, range(nBits, features.shape[1])].reshape(features.shape[0], features.shape[1]-nBits)
    elif set_features in ['fingerprint', 'f']:
        features = features[:, range(0, nBits)].reshape(features.shape[0], nBits)
        
    features, labels = drop_nan(features, labels)
    return features, labels

def get_data(logger, train_addr, test_addr, DUMMY, fingerprint, nBits, set_targets, set_features, random_state):
    if fingerprint in ['MACCS', 'maccs', 'Maccs', 'maccs (167)']:
        nBits = 167

    path = os.path.dirname(os.path.realpath(__file__)).replace("/src", "")
    if not os.path.exists(path+"/data/"):
        os.makedirs(path+"/data/")
    if not os.path.exists(path+"/data/preprocessed/"):
        os.makedirs(path+"/data/preprocessed/")

    logger.info("Train data")
    features, labels = processing(logger, train_addr, nBits, set_targets, set_features, DUMMY, fingerprint, path)
    if test_addr:
        logger.info("Test data")
        x_test, y_test = processing(logger, test_addr, nBits, set_targets, set_features, DUMMY, fingerprint, path)
        x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.8, stratify=labels, random_state=random_state)
    else:
        x_train, x, y_train, y = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=random_state)
        x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=0.8, stratify=y, random_state=random_state)

    logger.info("X_train: %s", str(x_train.shape))
    logger.info("Y_train: %s", str(y_train.shape))
    logger.info("X_test: %s", str(x_test.shape))
    logger.info("Y_test: %s", str(y_test.shape))
    logger.info("X_val: %s", str(x_val.shape))
    logger.info("Y_val: %s", str(y_val.shape))
    _, input_shape = x_train.shape
    _, output_shape = y_train.shape

    return x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape


if __name__ == "__main__":
    """
    Process dataset.
    """
    filename = os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/data/HIV.csv"
    fingerprint = 'morgan'
    nBits = 256
    DUMMY = False
    set_targets = [0]
    set_features = 'all'
    get_data(filename, DUMMY, fingerprint, nBits, set_targets, set_features)
