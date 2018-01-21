#!/usr/bin/env python

import os
import logging
import multiprocessing
import configparser
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from joblib import Parallel, delayed  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from src._desc_rdkit import smiles_to_desc_rdkit
from src.main import drop_nan, read_data_config


def create_physical(logger, smiles, verbose=5):    
    logger.info("Physic data extraction")    
    num_cores = multiprocessing.cpu_count()
    physic = np.array_split(pd.Series(smiles), num_cores)
    physic = np.array(physic)

    parallel = []
    parallel.append(Parallel(n_jobs=num_cores, verbose=5)(delayed(smiles_to_desc_rdkit)(pd.Series(p)) for (p) in physic))

    p_headers = parallel[0][0][0].columns.values.tolist()

    parallel = np.array(parallel)

    physic_data = parallel[0][0][0].T
    missing = np.array(parallel[0][0][1])
    for i in range(1, num_cores):
        physic_data = np.c_[physic_data, parallel[0][i][0].T]
        missing = np.append(missing, parallel[0][i][1])
        
    return missing, physic_data.T, p_headers
    

def featurization(logger, filename, DUMMY, fingerprint, n_bits, path, data_config):
    Dummy_n = 3000
    n_physical = 196
    if '.gz' in filename:
        data = pd.read_csv(filename, compression='gzip')
    else:
        data = pd.read_csv(filename)
    smiles = []
    if DUMMY:
        data = data[:Dummy_n]

    logger.info("Loading data")
    
    smiles = data["smiles"]
    data = data.drop("smiles", 1)
    if "mol_id" in list(data):
        data = data.drop("mol_id", 1)
        
    l_headers = list(data)
    
    missing, physic_data, p_headers = create_physical(logger, smiles)
    
    smiles = np.array(smiles)
    smiles = np.delete(smiles, missing)
    data = np.array(data)
    data = np.delete(data, missing, axis=0)
    #p_headers = "p" * physic_data.shape[1]
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
    
        
    filename_physical = filename.replace(".csv", "_physical.csv")
    head, _sep, tail = filename_physical.rpartition('/')
    filename_physical = path + "/data/preprocessed/physical/" + tail
        
    filename_labels = filename.replace(".csv", "_labels.csv")
    head, _sep, tail = filename_labels.rpartition('/')
    filename_labels = path + "/data/preprocessed/labels/" + tail

    fingerprints = pd.DataFrame(fingerprints)
    fingerprints.to_csv(filename_fingerprint+".gz", compression="gzip", sep=",")
    physical = pd.DataFrame(physic_data)
    physical.to_csv(filename_physical+".gz", compression="gzip", sep=",", header=p_headers)
    labels = pd.DataFrame(labels)
    labels.to_csv(filename_labels+".gz", compression="gzip", sep=",", header=l_headers)
    
        
    head, _sep, tail = filename.rpartition('/')
    name = tail.replace(".csv", "")
    
    if "_train" in name: name = "train"
    if "_test" in name: name = "test"
    if "_val" in name: name = "val"

    config = configparser.ConfigParser()
    config.read(data_config)
    
    config['DEFAULT']["labels_" + str(name)] = str(filename_labels.replace(path, '') + '.gz')
    config['DEFAULT']["physical_" + str(name)] = str(filename_physical.replace(path, '') + '.gz')
    try:
        config[str(fingerprint) + '_' + str(n_bits)]["fingerprint_" + str(name)] = str(filename_fingerprint.replace(path, '') + '.gz')
    except:
        with open(data_config, "a") as ini:
            ini.write('[' + str(fingerprint) + '_' + str(n_bits) + ']' + '\n')
        config.read(data_config)
        config[str(fingerprint) + '_' + str(n_bits)]["fingerprint_" + str(name)] = str(filename_fingerprint.replace(path, '') + '.gz')

    with open(data_config, 'w') as configfile:
        config.write(configfile)

    if DUMMY:
        fingerprints = np.array(fingerprints[:Dummy_n])
        physical = np.array(physical[:Dummy_n])
        labels = np.array(labels[:Dummy_n])
        
    return fingerprints, physical, labels
    
    
def download_data(labels_addr, physical_addr, fingerprint_addr):
    if '.gz' in labels_addr:
        labels = pd.read_csv(labels_addr, compression='gzip', index_col=0, header=0)
    else:
        labels = pd.read_csv(labels_addr)
        
    if '.gz' in physical_addr:
        physical = pd.read_csv(physical_addr, compression='gzip', index_col=0, header=0)
    else:
        physical = pd.read_csv(physical_addr)

    if '.gz' in fingerprint_addr:
        fingerprint = pd.read_csv(fingerprint_addr, compression='gzip', index_col=0, header=0)
    else:
        fingerprint = pd.read_csv(fingerprint_addr)
        
    return labels, physical, fingerprint
    
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

def preprocessing(logger, physical, fingerprints, labels):
    logger.info("Preprocessing")
    st = StandardScaler()
    remove_rows = []
    physical = np.array(physical)
    physical = physical.T
    for col in physical:
        for i, row in enumerate(col):
            if np.isnan(row):
                remove_rows.append(i)
            if not np.isfinite(row):
                remove_rows.append(i)

    physical = physical.T
    physical = pd.DataFrame(physical)
    physical.drop(physical.index[list(set(remove_rows))], inplace=True)
    #physical = np.array(physical)
    
    fingerprints.drop(fingerprints.index[list(set(remove_rows))], inplace=True)
    labels.drop(labels.index[list(set(remove_rows))], inplace=True)
    

    ################
    physical = normalize(physical)
    #features[n_bits:-85] = normalize(features[n_bits:-85])
    #features = normalize(features)
    #physical = physical.T
    return physical, fingerprints, labels
    #np.savetxt("out.csv", features, delimiter=",", fmt='%3f')
    
    
def get_data(logger, data_config, DUMMY, fingerprint, n_bits, set_targets, set_features, random_state, split):
    if fingerprint in ['MACCS', 'maccs', 'Maccs', 'maccs (167)']:
        n_bits = 167

    path = os.path.dirname(os.path.realpath(__file__)).replace("/src", "")
    if not os.path.exists(path+"/data/"):
        os.makedirs(path+"/data/")
    if not os.path.exists(path+"/data/preprocessed/"):
        os.makedirs(path+"/data/preprocessed/")
    if not os.path.exists(path+"/data/preprocessed/labels"):
        os.makedirs(path+"/data/preprocessed/labels")
    if not os.path.exists(path+"/data/preprocessed/morgan"):
        os.makedirs(path+"/data/preprocessed/morgan")
    if not os.path.exists(path+"/data/preprocessed/physical"):
        os.makedirs(path+"/data/preprocessed/physical")
        
    filename, filename_test, filename_val, labels_train, labels_test, labels_val, physical_train, physical_test, physical_val, fingerprint_train, fingerprint_test, fingerprint_val = read_data_config(data_config, str(fingerprint) + "_" + str(n_bits))

    logger.info("Train data")
    if fingerprint_train and physical_train and labels_train:       
        labels, physical, fingerprints = download_data(path+labels_train, path+physical_train, path+fingerprint_train)
        physical, fingerprints, labels = preprocessing(logger, physical, fingerprints, labels)
        x_train, y_train = compile_data(labels, physical, fingerprints, set_targets, set_features)
        x_train, y_train = drop_nan(x_train, y_train)
    else:
        fingerprints, physical, labels = featurization(logger, path+filename, DUMMY, fingerprint, n_bits, path, data_config)
        physical, fingerprints, labels = preprocessing(logger, physical, fingerprints, labels)
        x_train, y_train = compile_data(labels, physical, fingerprints, set_targets, set_features)
        x_train, y_train = drop_nan(x_train, y_train)

    logger.info("Test data") 
    if fingerprint_test and physical_test and labels_test:
        labels, physical, fingerprints = download_data(path+labels_test, path+physical_test, path+fingerprint_test)
        physical, fingerprints, labels = preprocessing(logger, physical, fingerprints, labels)
        x_test, y_test = compile_data(labels, physical, fingerprints, set_targets, set_features)
        x_test, y_test = drop_nan(x_test, y_test)
    elif filename_test:
        fingerprints, physical, labels = featurization(logger, path+filename_test, DUMMY, fingerprint, n_bits, path, data_config)
        physical, fingerprints, labels = preprocessing(logger, physical, fingerprints, labels)
        x_test, y_test = compile_data(labels, physical, fingerprints, set_targets, set_features)
        x_test, y_test = drop_nan(x_test, y_test)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=split, stratify=labels, random_state=random_state)
        
    logger.info("Val data") 
    if fingerprint_val and physical_val and labels_val:
        labels, physical, fingerprints = download_data(path+labels_val, path+physical_val, path+fingerprint_val)
        physical, fingerprints, labels = preprocessing(logger, physical, fingerprints, labels)
        x_val, y_val = compile_data(labels, physical, fingerprints, set_targets, set_features)
        x_val, y_val = drop_nan(x_val, y_val)
    elif filename_val:
        fingerprints, physical, labels = featurization(logger, path+filename_val, DUMMY, fingerprint, n_bits, path, data_config)
        physical, fingerprints, labels = preprocessing(logger, physical, fingerprints, labels)
        x_val, y_val = compile_data(labels, physical, fingerprints, set_targets, set_features)
        x_val, y_val = drop_nan(x_val, y_val)
    else:
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=split, stratify=labels, random_state=random_state)

    logger.info("X_train: %s", str(x_train.shape))
    logger.info("Y_train: %s", str(y_train.shape))
    logger.info("X_val: %s", str(x_val.shape))
    logger.info("Y_val: %s", str(y_val.shape))
    logger.info("X_test: %s", str(x_test.shape))
    logger.info("Y_test: %s", str(y_test.shape))
    _, input_shape = x_train.shape
    _, output_shape = y_train.shape
    
    ##############
    #np.savetxt("x_train "+set_features+".csv", x_train, delimiter=",", fmt='%3f')
    #np.savetxt("y_train "+set_features+".csv", y_train, delimiter=",", fmt='%3f')

    return x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape


if __name__ == "__main__":
    """
    Process dataset.
    """
    filename = os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/data/HIV.csv"
    fingerprint = 'morgan'
    n_bits = 256
    DUMMY = False
    set_targets = [0]
    set_features = 'all'
    get_data(filename, DUMMY, fingerprint, n_bits, set_targets, set_features)
