#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from moloi.descriptors.descriptors import descriptor_rdkit, descriptor_mordred, descriptor_maccs, descriptor_morgan, descriptor_spectrophore
from moloi.config_processing import read_data_config
from moloi.splits.scaffold_split import scaffold_split
from moloi.splits.cluster_split import cluster_split
import sys
if sys.version_info[0] == 2: # for python2
    import ConfigParser
else:
    import configparser as ConfigParser


def featurization(logger, filename, n_bits, path, data_config, verbose, descriptors, n_jobs, split_type):
    logger.info("Loading data")
    data = pd.read_csv(filename)
    # Cleaning
    if "mol_id" in list(data):
        data = data.drop("mol_id", 1)

    # Smiles
    smiles = data["smiles"]
    
    mordred_features, rdkit_features, maccs_fingerprints, morgan_fingerprints, spectrophore_fingerprints = False, False, False, False, False
    config = ConfigParser.ConfigParser()
    config.read(data_config)
    head, _sep, tail = filename.rpartition('/')
    name = tail.replace(".csv", "")
    
    if "_train" in name: name = "train"
    if "_test" in name: name = "test"
    if "_val" in name: name = "val"
    # Now it is necessary to calculate both physical descriptors, because they may lose the molecules that will affect the other descriptors and labels.
    missing = []

    logger.info("After processing shapes:")
    if 'mordred' in descriptors or not config.has_option(split_type, 'mordred_'+name):
        mordred_missing, mordred_features = descriptor_mordred(logger, smiles, verbose, n_jobs)
        
        mordred_address = filename.replace(".csv", "_mordred.csv")
        head, _sep, tail = mordred_address.rpartition('/')
        mordred_address = path + "/data/preprocessed/mordred/" + tail
        missing.append(mordred_missing)
        logger.info("mordred_features shape: %s", str(np.array(mordred_features).shape))

    if 'rdkit' in descriptors or not config.has_option(split_type, 'rdkit_'+name):
        rdkit_missing, rdkit_features = descriptor_rdkit(logger, smiles, verbose, n_jobs)
        
        rdkit_address = filename.replace(".csv", "_rdkit.csv")
        head, _sep, tail = rdkit_address.rpartition('/')
        rdkit_address = path + "/data/preprocessed/rdkit/" + tail
        missing.append(rdkit_missing)
        logger.info("rdkit_features shape: %s", str(np.array(rdkit_features).shape))

    smiles = np.array(smiles)
    missing = np.array(missing)
    if len(missing) > 1: # if missing in both rdkit and mordred
        missing = np.concatenate([missing[0],missing[1]])

    

    logger.info("After cleaning shapes:")
    
    if 'rdkit' in descriptors or not config.has_option(split_type, 'rdkit_'+name):
        rdkit_features = np.array(rdkit_features)
        rdkit_features = np.delete(rdkit_features, missing, axis=0)
        rdkit_features = pd.DataFrame(rdkit_features)
        rdkit_features.to_csv(rdkit_address+".gz", compression="gzip", sep=",", index=False)
        logger.info("rdkit_features shape: %s", str(np.array(rdkit_features).shape))
    
    if 'mordred' in descriptors or not config.has_option(split_type, 'mordred_'+name):
        mordred_features = np.array(mordred_features)
        mordred_features = np.delete(mordred_features, missing, axis=0)
        mordred_features = pd.DataFrame(mordred_features)
        mordred_features.to_csv(mordred_address+".gz", compression="gzip", sep=",", index=False)
        logger.info("mordred_features shape: %s", str(np.array(mordred_features).shape))

    smiles = np.delete(smiles, missing)

    labels = data.drop("smiles", 1)
    labels = np.array(labels)
    labels = np.delete(labels, missing, axis=0)
    
    if (labels.dtype != np.dtype('int64')) and (labels.dtype != np.dtype('int')) and (labels.dtype != np.dtype('float')) and (labels.dtype != np.dtype('float64')): # labels encoder if not ints
        l = []
        for i in range(labels.shape[1]):
            if isinstance(labels[i][0], str):
                le = LabelEncoder()
                le.fit(data[:,i])
                c = le.transform(labels[:,i])
                l.append(c)
            else:
                l.append(data[:, i])
        labels = np.array(l).T

    labels_address = filename.replace(".csv", "_labels.csv")
    head, _sep, tail = labels_address.rpartition('/')
    labels_address = path + "/data/preprocessed/labels/" + tail

    smiles = np.array(smiles)
    labels = np.array(labels)
    labels = np.c_[smiles, labels]
    labels = pd.DataFrame(labels)
    #labels.rename({'0': "labels"}, axis='columns')
    labels.to_csv(labels_address+".gz", compression="gzip", sep=",", index=False)

    for i in ['MACCS', 'maccs', 'Maccs', 'maccs (167)']:
        if i in descriptors:
            maccs_fingerprints = descriptor_maccs(logger, smiles)
            maccs_address = filename.replace(".csv", "_maccs.csv")
            head, _sep, tail = maccs_address.rpartition('/')
            maccs_address = path + "/data/preprocessed/maccs/" + tail
            maccs_fingerprints = pd.DataFrame(maccs_fingerprints)
            maccs_fingerprints.to_csv(maccs_address+".gz", compression="gzip", sep=",", index=False)
            logger.info("maccs_fingerprints shape: %s", str(np.array(maccs_fingerprints).shape))

    for i in ['MORGAN', 'Morgan', 'morgan', 'morgan (n)', 'ECFP']:
        if i in descriptors:
            morgan_fingerprints = descriptor_morgan(logger, smiles, n_bits)
            morgan_address = filename.replace(".csv", "_morgan_"+str(n_bits)+".csv")
            head, _sep, tail = morgan_address.rpartition('/')
            morgan_address = path + "/data/preprocessed/morgan/" + tail
            morgan_fingerprints = pd.DataFrame(morgan_fingerprints)
            morgan_fingerprints.to_csv(morgan_address+".gz", compression="gzip", sep=",", index=False)
            logger.info("morgan_fingerprints shape: %s", str(np.array(morgan_fingerprints).shape))

    for i in ['spectrophore']:
        if i in descriptors:
            spectrophore_fingerprints = descriptor_spectrophore(logger, smiles, n_bits)
            spectrophore_address = filename.replace(".csv", "_spectrophore_"+str(n_bits)+".csv")
            head, _sep, tail = spectrophore_address.rpartition('/')
            spectrophore_address = path + "/data/preprocessed/spectrophore/" + tail
            spectrophore_fingerprints = pd.DataFrame(spectrophore_fingerprints)
            spectrophore_fingerprints.to_csv(spectrophore_address+".gz", compression="gzip", sep=",", index=False)
            logger.info("spectrophore_fingerprints shape: %s", str(np.array(spectrophore_fingerprints).shape))

    # Filling config
    head, _sep, tail = filename.rpartition('/')
    name = tail.replace(".csv", "")
    
    if "_train" in name: name = "train"
    if "_test" in name: name = "test"
    if "_val" in name: name = "val"

    if split_type is False:
        split_type = 'DEFAULT'

    if split_type not in config.sections():
        split_type = 'DEFAULT'


    config[split_type]["labels_" + str(name)] = str(labels_address.replace(path, '') + '.gz')
    for i in ['mordred']:
        if i in descriptors:
            try:
                config[split_type]["mordred_" + str(name)] = str(mordred_address.replace(path, '') + '.gz')
            except:
                config.read(data_config)
                config[split_type]["mordred_" + str(name)] = str(mordred_address.replace(path, '') + '.gz')
        
    for i in ['rdkit']:
        if i in descriptors:
            try:
                config[split_type]["rdkit_" + str(name)] = str(rdkit_address.replace(path, '') + '.gz')
            except:
                config.read(data_config)
                config[split_type]["rdkit_" + str(name)] = str(rdkit_address.replace(path, '') + '.gz')
     
    for i in ['MACCS', 'maccs', 'Maccs', 'maccs (167)']:
        if i in descriptors:
            try:
                config[split_type]["maccs_" + str(name)] = str(maccs_address.replace(path, '') + '.gz')
            except:
                config.read(data_config)
                config[split_type]["maccs_" + str(name)] = str(maccs_address.replace(path, '') + '.gz')

    for i in ['MORGAN', 'Morgan', 'morgan', 'morgan (n)', 'ECFP']:
        if i in descriptors:
            try:
                config[split_type]["morgan_" + str(n_bits) + '_' + str(name)] = str(morgan_address.replace(path, '') + '.gz')
            except:
                config.read(data_config)
                config[split_type]["fmorgan_" + str(n_bits) + '_' + str(name)] = str(morgan_address.replace(path, '') + '.gz')

    for i in ['spectrophore']:
        if i in descriptors:
            try:
                config[split_type]["spectrophore_" + str(n_bits) + '_' + str(name)] = str(spectrophore_address.replace(path, '') + '.gz')
            except:
                config.read(data_config)
                config[split_type]["spectrophore_" + str(n_bits) + '_' + str(name)] = str(spectrophore_address.replace(path, '') + '.gz')

    with open(data_config, 'w') as configfile:
        config.write(configfile)

    return labels, mordred_features, rdkit_features, maccs_fingerprints, morgan_fingerprints, spectrophore_fingerprints

    
def compile_data(labels, mordred_features, rdkit_features, maccs_fingerprints, morgan_fingerprints, spectrophore_fingerprints, set_targets):
    labels = np.array(labels)
    x = False
    if mordred_features is not False:
        if x is not False:
            x = np.c_[x, np.array(mordred_features)]
        else:
            x = np.array(mordred_features)
    if rdkit_features is not False:
        if x is not False:
            x = np.c_[x, np.array(rdkit_features)]
        else:
            x = np.array(rdkit_features)
    if maccs_fingerprints is not False:
        if x is not False:
            x = np.c_[x, np.array(maccs_fingerprints)]
        else:
            x = np.array(maccs_fingerprints)
    if morgan_fingerprints is not False:
        if x is not False:
            x = np.c_[x, np.array(morgan_fingerprints)]
        else:
            x = np.array(morgan_fingerprints)
    if spectrophore_fingerprints is not False:
        if x is not False:
            x = np.c_[x, np.array(spectrophore_fingerprints)]
        else:
            x = np.array(spectrophore_fingerprints)
    if set_targets:
        y = labels[:, set_targets].reshape(labels.shape[0], len(set_targets))
    return x, y


def drop_nan(x, y):
    """
    Remove rows with NaN in data and labes relevant to this rows.
    """
    _, targ = x.shape
    table = np.c_[x, y]
    table = pd.DataFrame(table)
    table = table.dropna(axis=0, how='any')
    table = np.array(table)
    x = table[:, 0:targ]
    y = table[:, targ:]
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
    
    # Normalize
    transformer_X = StandardScaler().fit(x)
    x = transformer_X.transform(x)

    return x


def load_data(logger, path, filename, labels_addr, maccs_addr, morgan_addr, spectrophore_addr, mordred_addr, rdkit_addr, set_targets, n_bits, data_config, verbose, descriptors, n_jobs, split_type, do_featurization=True):
    x, y, smiles, labels, full_smiles = [], [], [], [], []
    t_descriptors = descriptors[:]
    t_descriptors.append('labels')
    labels, maccs, morgan, mordred, rdkit, spectrophore = False, False, False, False, False, False

    if labels_addr and os.path.isfile(path + labels_addr):
        labels = pd.read_csv(path + labels_addr, header=0)
        t_descriptors.remove('labels')

    for i in ['MACCS', 'maccs', 'Maccs', 'maccs (167)']:
        if i in t_descriptors:
            if maccs_addr and os.path.isfile(path + maccs_addr):
                maccs = pd.read_csv(path + maccs_addr, header=0)
                t_descriptors.remove(i)
    
    for i in ['MORGAN', 'Morgan', 'morgan', 'morgan (n)', 'ECFP']:
        if i in t_descriptors:
            if morgan_addr and os.path.isfile(path + morgan_addr):
                morgan = pd.read_csv(path + morgan_addr, header=0)
                t_descriptors.remove(i)
    
    for i in ['spectrophore']:
        if i in t_descriptors:
            if spectrophore_addr and os.path.isfile(path + spectrophore_addr):
                spectrophore = pd.read_csv(path + spectrophore_addr, header=0)
                t_descriptors.remove(i)
    
    if 'mordred' in t_descriptors:
        if mordred_addr and os.path.isfile(path + mordred_addr):
            mordred = pd.read_csv(path + mordred_addr, header=0)
            t_descriptors.remove('mordred')
    
    if 'rdkit' in t_descriptors:
        if rdkit_addr and os.path.isfile(path + rdkit_addr):
            rdkit = pd.read_csv(path + rdkit_addr, header=0)
            t_descriptors.remove('rdkit')

    if filename and len(t_descriptors) == 0:
        logger.info("Data loaded from config")
    elif filename:
        if os.path.isfile(path+filename):
            if do_featurization:
                t_labels, t_mordred, t_rdkit, t_maccs, t_morgan, t_spectrophore = featurization(logger, path+filename, n_bits, path, data_config, verbose, t_descriptors, n_jobs, split_type)
            else:
                t_labels, t_mordred, t_rdkit, t_maccs, t_morgan, t_spectrophore = [], [], [], [], [], []
            if labels is False:
                labels = t_labels
            if mordred is False:
                mordred = t_mordred
            if rdkit is False:
                rdkit = t_rdkit
            if maccs is False:
                maccs = t_maccs
            if morgan is False:
                morgan = t_morgan
            if spectrophore is False:
                spectrophore = t_spectrophore
        else:
            logger.info("Can not load data")
            return x, y, smiles, labels, full_smiles
    else:
        logger.info("Can not load data")
        return x, y, smiles, labels, full_smiles
    
    try:
        if labels is not False:
            #if 'smiles' in labels.columns:
            smiles = labels.iloc[:,0]
            labels = labels.iloc[:,1:]
        if mordred is not False:
            if 'smiles' in mordred.columns:
                mordred = mordred.drop("smiles", 1)
        if rdkit is not False:
            if 'smiles' in rdkit.columns:
                rdkit = rdkit.drop("smiles", 1)
        if maccs is not False:
            if 'smiles' in maccs.columns:
                maccs = maccs.drop("smiles", 1)
        if morgan is not False:
            if 'smiles' in morgan.columns:
                morgan = morgan.drop("smiles", 1)
        if spectrophore is not False:
            if 'smiles' in spectrophore.columns:
                spectrophore = spectrophore.drop("smiles", 1)

        full_smiles = pd.DataFrame(smiles)
        x, y = compile_data(labels, mordred, rdkit, maccs, morgan, spectrophore, set_targets)
        smiles = np.array(smiles)
        table = np.c_[x, y]
        table = np.c_[table, smiles]
        table = pd.DataFrame(table)
        table = table.dropna(axis=0, how='any')
        table = np.array(table)
        x = table[:, 0:x.shape[1]]
        y = table[:, x.shape[1]:(x.shape[1] + y.shape[1])]
        smiles = table[:, (x.shape[1] + y.shape[1]):]
        #smiles = smiles[:,0]
        smiles = smiles.reshape(-1,1)
        smiles = np.ravel(smiles)
        
    except UnboundLocalError:
        pass

    return x, y, smiles, labels, full_smiles


def split_test_val(split_type, x_train, y_train, smiles_train, labels_train, smiles_train_full, split_size, random_state):
    split_size=1-split_size*2
    x_train, y_train, x_test, y_test, smiles_train, smiles_test, labels_train, labels_test, smiles_train_full, smiles_test_full = split(split_type, x_train, y_train, smiles_train, labels_train, smiles_train_full, split_size, random_state)

    smiles_test = np.array(smiles_test)
    x_test = np.array(x_test)

    split_size=0.5
    x_test, y_test, x_val, y_val, smiles_test, smiles_val, labels_test, labels_val, smiles_test_full, smiles_val_full = split(split_type, x_test, y_test, smiles_test, labels_test, smiles_test_full, split_size, random_state)
    return x_test, y_test, x_train, y_train, x_val, y_val, smiles_test, smiles_train, smiles_val, labels_test, labels_train, labels_val, smiles_test_full, smiles_train_full, smiles_val_full


def split(split_type, x_train, y_train, smiles_train, labels_train, smiles_train_full, split_size, random_state):
    if split_type == "scaffold":
        train, test = scaffold_split(smiles_train, frac_train=1-split_size)
    elif split_type == "cluster":
        train, test = cluster_split(smiles_train, test_cluster_id=1, n_splits=2)
    elif split_type == "random":
        shuffle = True
        stratify = None
        idx = [i for i in range(len(smiles_train))]
        train, test = train_test_split(idx, test_size=split_size, stratify=stratify, shuffle=shuffle, random_state=random_state)
    elif split_type == "stratified":
        shuffle = True
        stratify = y_train
        idx = [i for i in range(len(smiles_train))]
        train, test = train_test_split(idx, test_size=split_size, stratify=stratify, shuffle=shuffle, random_state=random_state)
    else:
        print("Wrong split type")
        sys.exit(0)

    x_test = x_train[test]
    y_test = y_train[test]
    x_train = x_train[train]
    y_train = y_train[train]
    smiles_test = smiles_train[test]
    smiles_train = smiles_train[train]
    labels_test = labels_train.iloc[test]
    labels_train = labels_train.iloc[train]
    smiles_test_full = smiles_train_full.iloc[test]
    smiles_train_full = smiles_train_full.iloc[train]
    return x_test, y_test, x_train, y_train, smiles_test, smiles_train, labels_test, labels_train, smiles_test_full, smiles_train_full


def get_data(logger, data_config, n_bits, set_targets, random_state, split_type, split_size, verbose, descriptors, n_jobs):
    #if ['MACCS', 'maccs', 'Maccs', 'maccs (167)'] in descriptors:
    #    n_bits = 167 # constant for maccs fingerprint
    
    path = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
    dirs = ["/data/", "/data/preprocessed/", "/data/preprocessed/labels", "/data/preprocessed/morgan", "/data/preprocessed/maccs", "/data/preprocessed/spectrophore", "/data/preprocessed/rdkit", "/data/preprocessed/mordred"]
    for d in dirs:
        if not os.path.exists(path+d):
            os.makedirs(path+d)
    filename_train, filename_test, filename_val, labels_train, labels_test, labels_val, maccs_train, maccs_test, maccs_val, morgan_train, morgan_test, morgan_val, spectrophore_train, spectrophore_test, spectrophore_val, mordred_train, mordred_test, mordred_val, rdkit_train, rdkit_test, rdkit_val = read_data_config(data_config, descriptors, n_bits, split_type)
    
    logger.info("Load train data")
    x_train, y_train, smiles_train, labels_train, smiles_train_full = load_data(logger, path, filename_train, labels_train, maccs_train, morgan_train, spectrophore_train, mordred_train, rdkit_train, set_targets, n_bits, data_config, verbose, descriptors, n_jobs, split_type)
    logger.info("Load test data")
    x_test, y_test, smiles_test, labels_test, smiles_test_full = load_data(logger, path, filename_test, labels_test, maccs_test, morgan_test, spectrophore_test, mordred_test, rdkit_test, set_targets, n_bits, data_config, verbose, descriptors, n_jobs, split_type)
    logger.info("Load val data")
    x_val, y_val, smiles_val, labels_val, smiles_val_full = load_data(logger, path, filename_val, labels_val, maccs_val, morgan_val, spectrophore_val, mordred_val, rdkit_val, set_targets, n_bits, data_config, verbose, descriptors, n_jobs, split_type)
    
    logger.info("Loaded from config")
    logger.info("x_train shape: %s", str(np.array(x_train).shape))
    logger.info("x_test shape: %s", str(np.array(x_test).shape))
    logger.info("x_val shape: %s", str(np.array(x_val).shape))
    logger.info("y_train shape: %s", str(np.array(y_train).shape))
    logger.info("y_test shape: %s", str(np.array(y_test).shape))
    logger.info("y_val shape: %s", str(np.array(y_val).shape))

    if len(x_train) > 1 and len(x_test) > 1 and len(x_val) > 1 and len(y_train) > 1 and len(y_test) > 1 and len(y_val) > 1:
        split_type = False

    #if split_type == "scaffold":
    if split_type:
        for i in range(100):
            count = 0
            if (len(x_test) < 1 or len(y_test) < 1) and (len(x_val) < 1 or len(y_val) < 1):
                logger.info("Split test and val data")
                x_test, y_test, x_train, y_train, x_val, y_val, smiles_test, smiles_train, smiles_val, labels_test, labels_train, labels_val, smiles_test_full, smiles_train_full, smiles_val_full = split_test_val(split_type, x_train, y_train, smiles_train, labels_train, smiles_train_full, split_size, random_state)
            
            elif len(x_test) < 1 or len(y_test) < 1:
                logger.info("Split test data")
                x_test, y_test, x_train, y_train, smiles_test, smiles_train, labels_test, labels_train, smiles_test_full, smiles_train_full = split(split_type, x_train, y_train, smiles_train, labels_train, smiles_train_full, split_size, random_state)
            
            elif len(x_val) < 1 or len(y_val) < 1:
                logger.info("Split val data")
                x_train, y_train, x_val, y_val, smiles_train, smiles_val, labels_train, labels_val, smiles_train_full, smiles_val_full = split(split_type, x_train, y_train, smiles_train, labels_train, smiles_train_full, split_size, random_state)

            for j in [y_train, y_val, y_test]:
                if len(np.unique(j)) > 1:
                    count += 1

            if count == 3:
                break
            random_state += 1

        if count != 3:
            logger.info("Can not create a good data split.")
            sys.exit(0)

    if split_type:
        names = ["smiles"] + ["label"]*labels_train.shape[1]       
        train = pd.DataFrame(np.c_[smiles_train_full, labels_train], columns=names)
        test = pd.DataFrame(np.c_[smiles_test_full, labels_test], columns=names)
        val = pd.DataFrame(np.c_[smiles_val_full, labels_val], columns=names)
        path = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
        train.to_csv(path + filename_train.replace("_train.csv","_"+split_type+"_train.csv"), sep=",", index=False)
        test.to_csv(path + filename_train.replace("_train.csv","_"+split_type+"_test.csv"), sep=",", index=False)
        val.to_csv(path + filename_train.replace("_train.csv","_"+split_type+"_val.csv"), sep=",", index=False)

        files_config = ConfigParser.ConfigParser()

        files_config.read(data_config)
        name = os.path.basename(filename_train).replace(".csv","")
        try:
            files_config[split_type]["dataset_train"] = filename_train.replace("_train.csv","_"+split_type+"_train.csv")
            files_config[split_type]["dataset_test"] = filename_train.replace("_train.csv","_"+split_type+"_test.csv")
            files_config[split_type]["dataset_val"] = filename_train.replace("_train.csv","_"+split_type+"_val.csv")
        except:
            with open(data_config, "a") as ini:
                ini.write('[' + split_type + ']')
            files_config.read(data_config)
            files_config[split_type]["dataset_train"] = filename_train.replace("_train.csv","_"+split_type+"_train.csv")
            files_config[split_type]["dataset_test"] = filename_train.replace("_train.csv","_"+split_type+"_test.csv")
            files_config[split_type]["dataset_val"] = filename_train.replace("_train.csv","_"+split_type+"_val.csv")
        with open(data_config, 'w') as configfile:
            files_config.write(configfile)
    
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

    return x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape, smiles_train


if __name__ == "__main__":
    """
    Process dataset.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    data_config = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/data/data_configs/bace.ini"
    fingerprint = 'morgan'
    descriptor = 'mordred'
    n_bits = 256
    set_targets = [0]
    split = 0.2
    random_state = 13
    get_data(logger, data_config, fingerprint, n_bits, set_targets, random_state, split, verbose, descriptor, False)