#!/usr/bin/env python

import os
import logging
import multiprocessing
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from joblib import Parallel, delayed  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from src._desc_rdkit import smiles_to_desc_rdkit, smiles_to_desc_mordred
from src.main import drop_nan, read_data_config

import sys
if sys.version_info[0] == 2:
    import ConfigParser
else:
    import configparser


def create_physical(logger, smiles, verbose, descriptor):    
    logger.info("Physic data extraction")    
    
    if descriptor == "rdkit":
        num_cores = multiprocessing.cpu_count()
        physic = np.array_split(pd.Series(smiles), num_cores)
        physic = np.array(physic)

        parallel = []
        parallel.append(Parallel(n_jobs=num_cores, verbose=verbose)(delayed(smiles_to_desc_rdkit)(pd.Series(p)) for (p) in physic))

        p_headers = parallel[0][0][0].columns.values.tolist()

        parallel = np.array(parallel)

        physic_data = parallel[0][0][0].T
        missing = np.array(parallel[0][0][1])
        for i in range(1, num_cores):
            physic_data = np.c_[physic_data, parallel[0][i][0].T]
            missing = np.append(missing, parallel[0][i][1])
        physic_data = pd.DataFrame(physic_data)
        physic_data = physic_data.T
    
    elif descriptor == "mordred":
        physic_data, missing = smiles_to_desc_mordred(pd.Series(smiles))
        #physic_data = pd.DataFrame(physic_data)
        
    return missing, physic_data
    

def featurization(logger, filename, fingerprint, n_bits, path, data_config, verbose, descriptor):
    data = pd.read_csv(filename)
    smiles = []

    logger.info("Loading data")
    
    smiles = data["smiles"]
    data = data.drop("smiles", 1)
    if "mol_id" in list(data):
        data = data.drop("mol_id", 1)
        
    l_headers = list(data)
    
    missing, physic_data = create_physical(logger, smiles, verbose, descriptor)
    n_physical = physic_data.shape[1]
    
    smiles = np.array(smiles)
    smiles = np.delete(smiles, missing)
    data = np.array(data)
    data = np.delete(data, missing, axis=0)
    _, cols = data.shape

    labels = data    
    if (data.dtype != np.dtype('int64')) and (data.dtype != np.dtype('int')):
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

    #labels = np.delete(labels, missing, axis=0)

    logger.info("Featurization")
    ms = [Chem.MolFromSmiles(x) for x in smiles]
    if fingerprint in ['MACCS', 'maccs', 'Maccs', 'maccs (167)']:
        features = [MACCSkeys.GenMACCSKeys(x) for x in ms if x]
        fingerprints = np.array(features)
        #features = np.c_[features, physic_data]
        #featurized = np.c_[features, labels]
        filename_fingerprint = filename.replace(".csv", "_maccs.csv")

    if fingerprint in ['MORGAN', 'Morgan', 'morgan', 'morgan (n)']:
        features = [AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=n_bits) for x in ms if x]
        fingerprints = np.array(features)
        #features = np.c_[features, physic_data]
        #physical = np.c_[physic_data, labels]
        #featurized = np.c_[features, labels]
        filename_fingerprint = filename.replace(".csv", "_morgan_"+str(n_bits)+".csv")
        
    head, _sep, tail = filename_fingerprint.rpartition('/')
    filename_fingerprint = path + "/data/preprocessed/morgan/" + tail
    
        
    filename_physical = filename.replace(".csv", "_" + descriptor + "_physical.csv")
    head, _sep, tail = filename_physical.rpartition('/')
    filename_physical = path + "/data/preprocessed/" + descriptor + "/" + tail
        
    filename_labels = filename.replace(".csv", "_labels.csv")
    head, _sep, tail = filename_labels.rpartition('/')
    filename_labels = path + "/data/preprocessed/labels/" + tail

    fingerprints = pd.DataFrame(fingerprints)
    fingerprints.to_csv(filename_fingerprint+".gz", compression="gzip", sep=",")
    #physical = pd.DataFrame(physic_data)
    
    physic_data.to_csv(filename_physical+".gz", compression="gzip", sep=",")
    labels = pd.DataFrame(labels)
    labels.to_csv(filename_labels+".gz", compression="gzip", sep=",")

    head, _sep, tail = filename.rpartition('/')
    name = tail.replace(".csv", "")
    
    if "_train" in name: name = "train"
    if "_test" in name: name = "test"
    if "_val" in name: name = "val"

    if sys.version_info[0] == 2:
        config = ConfigParser.ConfigParser()
    else:
        config = configparser.ConfigParser()

    config.read(data_config)
    
    config['DEFAULT']["labels_" + str(name)] = str(filename_labels.replace(path, '') + '.gz')

    try:
        config[descriptor]["physical_" + str(name)] = str(filename_physical.replace(path, '') + '.gz')
    except:
        with open(data_config, "a") as ini:
            ini.write('[' + descriptor + ']' + '\n')
        config.read(data_config)
        config[descriptor]["physical_" + str(name)] = str(filename_physical.replace(path, '') + '.gz')
        
        
    
    try:
        config[str(fingerprint) + '_' + str(n_bits)]["fingerprint_" + str(name)] = str(filename_fingerprint.replace(path, '') + '.gz')
    except:
        with open(data_config, "a") as ini:
            ini.write('[' + str(fingerprint) + '_' + str(n_bits) + ']' + '\n')
        config.read(data_config)
        config[str(fingerprint) + '_' + str(n_bits)]["fingerprint_" + str(name)] = str(filename_fingerprint.replace(path, '') + '.gz')

    with open(data_config, 'w') as configfile:
        config.write(configfile)
        
    return fingerprints, physic_data, labels

    
def compile_data(labels, physical, fingerprint, set_targets, set_features):
    labels = np.array(labels)
    physical = np.array(physical)
    fingerprint = np.array(fingerprint)
        
    if set_targets:
        y = labels[:, set_targets].reshape(labels.shape[0], len(set_targets))
    if set_features in ['physical', 'p']:
        x = physical
    elif set_features in ['fingerprint', 'f']:
        x = fingerprint
    elif set_features in ['all', 'a']:
        x = np.c_[fingerprint, physical] 
    return x, y


def preprocessing(x):
    x = np.array(x)

    for j, col in enumerate(x):
        for i, row in enumerate(col):
            if not isinstance(row, (int, float, np.int64, np.float64)):
                x[j][i] = 0
            if x[j][i] == np.NaN: # do NOT remove
                x[j][i] = 0
            if not np.isfinite(x[j][i]): # do NOT remove
                x[j][i] = 0
                
    x = normalize(x)
    return x


def load_data(logger, path, filename, fingerprint_addr, physical_addr, labels_addr, set_targets, set_features, fingerprint, n_bits, data_config, verbose, descriptor):  
    x, y = [], []
    
    if fingerprint_addr and physical_addr and labels_addr and os.path.isfile(path+fingerprint_addr) and os.path.isfile(path+physical_addr) and os.path.isfile(path+labels_addr):
        fingerprints = pd.read_csv(path+fingerprint_addr,index_col=0, header=0)
        physical = pd.read_csv(path+physical_addr,index_col=0, header=0)
        labels  = pd.read_csv(path+labels_addr,index_col=0, header=0)
        
        #physical, fingerprints, labels = preprocessing(physical), preprocessing(fingerprints), preprocessing(labels)
        x, y = compile_data(labels, physical, fingerprints, set_targets, set_features)
        x, y = drop_nan(x, y) # becouse labels can be NaN in column

    elif filename:
        if os.path.isfile(path+filename):
            fingerprints, physical, labels = featurization(logger, path+filename, fingerprint, n_bits, path, data_config, verbose, descriptor)
            
            #physical, fingerprints, labels = preprocessing(physical), preprocessing(fingerprints), preprocessing(labels)
            x, y = compile_data(labels, physical, fingerprints, set_targets, set_features)
            x, y = drop_nan(x, y)

    else:
        print("Can not load data")
        
    #print('f',fingerprints.shape)
    #print('p',physical.shape)
    #print('l',labels.shape)
    
    return x, y

    
def get_data(logger, data_config, fingerprint, n_bits, set_targets, set_features, random_state, split_type, split_size, verbose, descriptor):
    if fingerprint in ['MACCS', 'maccs', 'Maccs', 'maccs (167)']:
        n_bits = 167 # constant for maccs fingerprint
    
    path = os.path.dirname(os.path.realpath(__file__)).replace("/src", "")
    dirs = ["/data/", "/data/preprocessed/", "/data/preprocessed/labels", "/data/preprocessed/morgan", "/data/preprocessed/rdkit", "/data/preprocessed/mordred"]
    for d in dirs:
        if not os.path.exists(path+d):
            os.makedirs(path+d)

    filename_train, filename_test, filename_val, labels_train, labels_test, labels_val, physical_train, physical_test, physical_val, fingerprint_train, fingerprint_test, fingerprint_val = read_data_config(data_config, str(fingerprint) + "_" + str(n_bits), descriptor)
    logger.info("Load train data")
    x_train, y_train = load_data(logger, path, filename_train, fingerprint_train, physical_train, labels_train, set_targets, set_features, fingerprint, n_bits, data_config, verbose, descriptor)
    logger.info("Load test data")
    x_test, y_test = load_data(logger, path, filename_test, fingerprint_test, physical_test, labels_test, set_targets, set_features, fingerprint, n_bits, data_config, verbose, descriptor)
    logger.info("Load val data")
    x_val, y_val = load_data(logger, path, filename_val, fingerprint_val, physical_val, labels_val, set_targets, set_features, fingerprint, n_bits, data_config, verbose, descriptor)

    if split_type == "random":
        if (len(x_test) < 1 or len(y_test) < 1) and (len(x_val) < 1 or len(y_val) < 1):
            logger.info("Split test and val data")
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=split_size*2, random_state=random_state, shuffle=True)
            x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=random_state, shuffle=True)
        
        elif len(x_test) < 1 or len(y_test) < 1:
            logger.info("Split test data")
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=split_size, random_state=random_state, shuffle=True)
        
        elif len(x_val) < 1 or len(y_val) < 1:
            logger.info("Split val data")
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=split_size, random_state=random_state, shuffle=True)

    elif split_type == "stratified":
        if (len(x_test) < 1 or len(y_test) < 1) and (len(x_val) < 1 or len(y_val) < 1):
            logger.info("Split test and val data")
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=split_size*2, stratify=y_train, random_state=random_state)
            x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test, random_state=random_state)
        
        elif len(x_test) < 1 or len(y_test) < 1:
            logger.info("Split test data")
            x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=split_size, stratify=y_train, random_state=random_state)
        
        elif len(x_val) < 1 or len(y_val) < 1:
            logger.info("Split val data")
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=split_size, stratify=y_train, random_state=random_state)
    
    x_train = preprocessing(x_train)
    x_test = preprocessing(x_test)
    x_val = preprocessing(x_val)
    
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    
    logger.info("X_train: %s", str(x_train.shape))
    logger.info("Y_train: %s", str(y_train.shape))
    logger.info("X_val: %s", str(x_val.shape))
    logger.info("Y_val: %s", str(y_val.shape))
    logger.info("X_test: %s", str(x_test.shape))
    logger.info("Y_test: %s", str(y_test.shape))

    _, input_shape = x_train.shape
    _, output_shape = y_train.shape

    return x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape


if __name__ == "__main__":
    """
    Process dataset.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    data_config = os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/data/data_configs/bace.ini"
    fingerprint = 'morgan'
    descriptor = 'mordred'
    n_bits = 256
    set_targets = [0]
    set_features = 'all'
    split = 0.2
    random_state = 13
    get_data(logger, data_config, fingerprint, n_bits, set_targets, set_features, random_state, split, verbose, descriptor)
