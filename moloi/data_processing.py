#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from moloi.descriptors.descriptors import descriptor_rdkit, descriptor_mordred
from moloi.descriptors.descriptors import descriptor_maccs, descriptor_morgan, descriptor_spectrophore
from moloi.descriptors.mordred_descriptor import mordred_fetures_names
from moloi.descriptors.rdkit_descriptor import rdkit_fetures_names
from moloi.config_processing import read_data_config
from moloi.splits.scaffold_split import scaffold_split
from moloi.splits.cluster_split import cluster_split
# from moloi.splitted_dataset import saving_splitted_dataset
from moloi.dictionaries import data_dict
import sys
if sys.version_info[0] == 2:  # for Python2
    import ConfigParser
else:
    import configparser as ConfigParser


def get_data(logger, options, random_state, verbose):
    # if ['MACCS', 'maccs', 'Maccs', 'maccs (167)'] in descriptors:
    #     n_bits = 167 # constant for maccs fingerprint

    path = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
    dirs = ["/data/", "/data/preprocessed/", "/data/preprocessed/labels",
            "/data/preprocessed/external", "/data/preprocessed/morgan",
            "/data/preprocessed/maccs", "/data/preprocessed/spectrophore",
            "/data/preprocessed/rdkit", "/data/preprocessed/mordred"]  # save outputs
    for d in dirs:
        if not os.path.exists(path+d):
            os.makedirs(path+d)

    data = data_dict()
    addresses = read_data_config(options.data_config, options.n_bits, options.split_type, options.split_s)
    logger.info("Load data")
    data = load_data(logger, path, addresses, options, data, verbose)

    logger.info("Loaded from config")
    logger.info("x_train shape: %s", str(np.array(data["x_train"]).shape))
    logger.info("x_test shape: %s", str(np.array(data["x_test"]).shape))
    logger.info("x_val shape: %s", str(np.array(data["x_val"]).shape))
    logger.info("y_train shape: %s", str(np.array(data["y_train"]).shape))
    logger.info("y_test shape: %s", str(np.array(data["y_test"]).shape))
    logger.info("y_val shape: %s", str(np.array(data["y_val"]).shape))

    print(data["y_train"])

    if (data["x_train"] is not False and data["x_test"] is not False and data["x_val"] is not False and
        data["y_train"] is not False and data["y_test"] is not False and data["y_val"] is not False):
        options.split_type = False

    # if split_type == "scaffold":
    if options.split_type:
        for i in range(100):
            count = 0
            if (data["x_test"] is False or data["y_test"] is False) and (data["x_val"] is False or data["y_val"] is False):
                logger.info("Split test and val data")
                data = split_test_val(options.split_type, data, options.split_s, random_state)

            elif data["x_test"] is False or data["y_test"] is False:
                logger.info("Split test data")
                data = split(options.split_type, data, 1-options.split_s, random_state, 'test')

            elif data["x_val"] is False or data["y_val"] is False:
                logger.info("Split val data")
                data = split(options.split_type, data, 1-options.split_s, random_state, 'val')

            for j in [data["y_train"], data["y_val"], data["y_test"]]:
                if len(np.unique(j)) > 1:
                    count += 1

            if count == 3:
                break
            random_state += 1

        if count != 3:
            logger.info("Can not create a good data split.")
            sys.exit(0)

################################################
# WARNING! Disable saving of splitted datasets #
################################################
#    if options.split_type:
#        head, _sep, tail = addresses["filename_train"].rpartition('/')
#        name = tail.replace(".csv", "").replace("_train","").replace("_test","").replace("_val","")
#        addresses["filename_train"] = name + "_train"
#        addresses["filename_test"] = name + "_test"
#        addresses["filename_val"] = name + "_val"
#
#
#        maccs_train, morgan_train, mordred_train, rdkit_train, spectrophore_train, external_train = split_x(x_train, options.n_bits, options.descriptors)
#        maccs_test, morgan_test, mordred_test, rdkit_test, spectrophore_test, external_test = split_x(x_test, options.n_bits, options.descriptors)
#        maccs_val, morgan_val, mordred_val, rdkit_val, spectrophore_val, external_val = split_x(x_val, options.n_bits, options.descriptors)

#       saving_splitted_dataset(path, labels_train, labels_test, labels_val, smiles_train, smiles_test, smiles_val, filename_train, filename_test, filename_val, maccs_train, maccs_test, maccs_val, morgan_train, morgan_test, morgan_val, mordred_train, mordred_test, mordred_val, rdkit_train, rdkit_test, rdkit_val, spectrophore_train, spectrophore_test, spectrophore_val, external_train, external_test, external_val, options)
################################################

    data["y_train"] = np.asarray(data["y_train"], dtype=int)
    data["y_test"] = np.asarray(data["y_test"], dtype=int)
    data["y_val"] = np.asarray(data["y_val"], dtype=int)

    print("")
    logger.info("Splitted data:")
    logger.info("X_train: %s", str(data["x_train"].shape))
    logger.info("Y_train: %s", str(data["y_train"].shape))
    logger.info("X_val: %s", str(data["x_val"].shape))
    logger.info("Y_val: %s", str(data["y_val"].shape))
    logger.info("X_test: %s", str(data["x_test"].shape))
    logger.info("Y_test: %s", str(data["y_test"].shape))

    return data


def load_data(logger, path, addresses, options, data, verbose, do_featurization=True):
    t_descriptors = options.descriptors[:]
    t_descriptors.append('labels')

    for j in ['_train', '_test', '_val']:
        X = {
            "labels": False,
            "maccs": False,
            "morgan": False,
            "mordred": False,
            "rdkit": False,
            "spectrophore": False,
            "external": False
            }

        X_t = dict(X)

        for i in ['labels', 'maccs', 'morgan', 'mordred', 'rdkit', 'spectrophore', 'external']:
            if i in t_descriptors:
                if addresses[i + j] and os.path.isfile(path + addresses[i + j]):
                    X[i] = pd.read_csv(path + addresses[i + j], header=0)
                    logger.info(i + " loaded from config")
                    t_descriptors.remove(i)

        if addresses["dataset" + j] and len(t_descriptors) == 0:
            logger.info("Data loaded from config")

        elif addresses["dataset" + j]:
            if os.path.isfile(path + addresses["dataset" + j]):
                if do_featurization:
                    X_t, addresses = featurization(logger, path + addresses["dataset" + j],
                                                   options, path, t_descriptors, X_t, addresses, j, verbose)

                for i in ['labels', 'maccs', 'morgan', 'mordred', 'rdkit', 'spectrophore', 'external']:
                    if X[i] is False and (i in options.descriptors or i in 'labels'):
                        X[i] = X_t[i]
            else:
                logger.info("Can not load data")
                sys.exit(0)
        else:
            logger.info("Can not load data")
            return data

        if 'smiles' in X['labels'].columns:
            data['smiles' + j] = X['labels']['smiles']
            data['labels' + j] = X['labels'].drop("smiles", 1)
        else:
            if isinstance(X['labels'].iloc[0, 0], str):
                data['smiles' + j] = X['labels'].iloc[:, 0]
                data['labels' + j] = X['labels'].iloc[:, 1:]
                X['smiles'] = data['smiles' + j]
                X['labels'] = data['labels' + j]
            else:
                data['smiles' + j] = X['labels'].iloc[:, 1]
                data['labels' + j] = X['labels'].iloc[:, 0]
                X['smiles'] = data['smiles' + j]
                X['labels'] = data['labels' + j]

        for i in ['maccs', 'morgan', 'mordred', 'rdkit', 'spectrophore', 'external']:
            if X[i] is not False:
                if 'smiles' in X[i].columns:
                    data[i + j] = X[i].drop("smiles", 1)

        if j in '_train':
            data['full_smiles' + j] = pd.DataFrame(data['smiles' + j])

        #####################
        # Select features here
        #####################
        data['x' + j], data['y' + j] = compile_data(X, options.targets)

        data['smiles' + j] = np.array(data['smiles' + j])
        if data['x' + j].shape[0] != data['y' + j].shape[0]:
            print("X amd Y are not match")
            print(data['x' + j].shape)
            print(data['y' + j].shape)
            sys.exit(0)
        table = np.c_[data['x' + j], data['y' + j]]

        table = np.c_[table, data['smiles' + j]]
        table = pd.DataFrame(table)
        table = table.dropna(axis=0, how='any')
        table = np.array(table)
        data['x' + j] = table[:, 0:data['x' + j].shape[1]]
        data['y' + j] = table[:, data['x' + j].shape[1]:(data['x' + j].shape[1] + data['y' + j].shape[1])]
        data['smiles' + j] = table[:, (data['x' + j].shape[1] + data['y' + j].shape[1]):]
        data['smiles' + j] = data['smiles' + j].reshape(-1, 1)
        data['smiles' + j] = np.ravel(data['smiles' + j])

        # except UnboundLocalError:
        #     pass
        return data


def compile_data(X, set_targets):
    X['labels'] = np.array(X['labels'])
    x = False
    for i in ['labels', 'maccs', 'morgan', 'mordred', 'rdkit', 'spectrophore', 'external']:
        if X[i] is not False:
            print(i, X[i].shape)

            if x is not False:
                x = np.c_[x, np.array(X[i])]
            else:
                x = np.array(X[i])

    if set_targets:
        y = X['labels'][:, set_targets].reshape(-1, len(set_targets))
    return x, y


def featurization(logger, filename, options, path, descriptors, X_t, addresses, j, verbose):
    logger.info("Loading data")
    logger.info("Filename: %s", filename)
    logger.info("Descriptors: %s", str(descriptors))
    data = pd.read_csv(filename)
    # Cleaning
    if "mol_id" in list(data):
        data = data.drop("mol_id", 1)

    # Smiles
    try:
        smiles = data["smiles"]
        data = data.drop("smiles", 1)
    except:
        smiles = data["0"]
        data = data.drop("0", 1)

    config = ConfigParser.ConfigParser()
    config.read(options.data_config)

    # Now it is necessary to calculate both physical descriptors,
    # because they may lose the molecules that will affect the other descriptors and labels.
    missing = []
    section = options.split_type + " " + str(options.split_s)

    logger.info("After processing shapes:")
    if 'mordred' in descriptors and not config.has_option(section, 'mordred' + j):
        mordred_missing, mordred_features = descriptor_mordred(logger, smiles, verbose, options.n_jobs)

        addresses["mordred"+j] = filename.replace(".csv", "_mordred.csv")
        head, _sep, tail = addresses["mordred"+j].rpartition('/')
        addresses["mordred"+j] = path + "/data/preprocessed/mordred/" + tail
        missing.append(mordred_missing)
        logger.info("mordred_features shape: %s", str(np.array(mordred_features).shape))

    if 'rdkit' in descriptors and not config.has_option(section, 'rdkit' + j):
        rdkit_missing, rdkit_features = descriptor_rdkit(logger, smiles, verbose, options.n_jobs)

        addresses["rdkit"+j] = filename.replace(".csv", "_rdkit.csv")
        head, _sep, tail = addresses["rdkit"+j].rpartition('/')
        addresses["rdkit"+j] = path + "/data/preprocessed/rdkit/" + tail
        missing.append(rdkit_missing)
        logger.info("rdkit_features shape: %s", str(np.array(rdkit_features).shape))

    smiles = np.array(smiles)
    missing = np.array(missing)
    if len(missing) > 1:  # if missing in both rdkit and mordred
        missing = np.concatenate([missing[0], missing[1]])

    logger.info("After cleaning shapes:")

    if 'rdkit' in descriptors and not config.has_option(section, 'rdkit'+j):
        rdkit_features = np.array(rdkit_features)
        rdkit_features = np.delete(rdkit_features, missing, axis=0)
        rdkit_features = pd.DataFrame(rdkit_features)
        rdkit_features.columns = rdkit_fetures_names()
        rdkit_features.to_csv((addresses["rdkit"+j]+".gz").replace('.gz.gz', '.gz'), compression="gzip", sep=",", index=False)
        logger.info("rdkit_features shape: %s", str(np.array(rdkit_features).shape))

        X_t["rdkit"] = rdkit_features

    if 'mordred' in descriptors and not config.has_option(section, 'mordred'+j):
        mordred_features = np.array(mordred_features)
        mordred_features = np.delete(mordred_features, missing, axis=0)
        mordred_features = pd.DataFrame(mordred_features)
        mordred_features.columns = mordred_fetures_names()
        mordred_features.to_csv((addresses["mordred"+j]+".gz").replace('.gz.gz', '.gz'),
                                compression="gzip", sep=",", index=False)
        logger.info("mordred_features shape: %s", str(np.array(mordred_features).shape))

        X_t["mordred"] = mordred_features

    smiles = np.delete(smiles, missing)
    labels = data
    labels = np.array(labels)
    labels = np.delete(labels, missing, axis=0)

    if ((labels.dtype != np.dtype('int64')) and (labels.dtype != np.dtype('int')) and
       (labels.dtype != np.dtype('float')) and (labels.dtype != np.dtype('float64'))):  # labels encoder if not ints
        lb = []
        le = LabelEncoder()
        le.fit(labels[:, 0])
        for i in range(labels.shape[1]):
            if isinstance(labels[i][0], str):
                c = le.transform(labels[:, i])
                lb.append(c)
            else:
                lb.append(labels[:, 0])
        labels = np.array(lb).T

    if 'labels' not in filename:
        labels_address = filename.replace(".csv", "_labels.csv")
    else:
        labels_address = filename
    head, _sep, tail = labels_address.rpartition('/')
    labels_address = path + "/data/preprocessed/labels/" + tail
    # labels_address.replace(".csv", split_type+'_'+split_size+'.csv')

    smiles = np.array(smiles)
    labels = np.array(labels)
    labels = np.c_[smiles, labels]
    labels = pd.DataFrame(labels)
    # labels.rename({'0': "labels"}, axis='columns')
    labels.to_csv((labels_address+".gz").replace('.gz.gz', '.gz'), compression="gzip", sep=",", index=False)
    X_t["labels"] = labels
    addresses["labels" + j] = labels_address

    for i in ['maccs']:
        if i in descriptors:
            X_t[i] = descriptor_maccs(logger, smiles)
            addresses[i + j] = filename.replace(".csv", "_maccs.csv")
            head, _sep, tail = addresses[i + j].rpartition('/')
            addresses[i + j] = path + "/data/preprocessed/maccs/" + tail
            X_t[i] = pd.DataFrame(X_t[i])
            X_t[i].to_csv((addresses[i + j]+".gz").replace('.gz.gz', '.gz'),
                          compression="gzip", sep=",", index=False)
            logger.info(i + " fingerprints shape: %s", str(np.array(X_t[i]).shape))

    for i in ['morgan']:
        if i in descriptors:
            X_t[i] = descriptor_morgan(logger, smiles, options.n_bits)
            addresses[i + j] = filename.replace(".csv", "_morgan_"+str(options.n_bits)+".csv")
            head, _sep, tail = addresses[i + j].rpartition('/')
            addresses[i + j] = path + "/data/preprocessed/morgan/" + tail
            X_t[i] = pd.DataFrame(X_t[i])
            X_t[i].to_csv((addresses[i + j]+".gz").replace('.gz.gz', '.gz'),
                          compression="gzip", sep=",", index=False)
            logger.info(i + " fingerprints shape: %s", str(np.array(X_t[i]).shape))

    for i in ['spectrophore']:
        if i in descriptors:
            X_t[i] = descriptor_spectrophore(logger, smiles, options.n_bits, options.n_jobs, verbose)
            addresses[i + j] = filename.replace(".csv", "_spectrophore_"+str(options.n_bits)+".csv")
            head, _sep, tail = addresses[i + j].rpartition('/')
            addresses[i + j] = path + "/data/preprocessed/spectrophore/" + tail
            X_t[i] = pd.DataFrame(X_t[i])
            X_t[i].to_csv((addresses[i + j]+".gz").replace('.gz.gz', '.gz'),
                          compression="gzip", sep=",", index=False)
            logger.info(i + " fingerprints shape: %s", str(np.array(X_t[i]).shape))

    filling_config(path, options, j, descriptors, addresses)
    return X_t, addresses


def filling_config(path, options, j, descriptors, addresses):
    config = ConfigParser.ConfigParser()
    config.read(options.data_config)

    if options.split_type is False:
        section = 'init'
    else:
        section = options.split_type+" "+str(options.split_s)

    if options.split_type+" "+str(options.split_s) not in config.sections():
        section = 'init'
    else:
        section = options.split_type+" "+str(options.split_s)

    # WARNING! Rewrite original dataset address in config
    if addresses['labels' + j]:
        config[section]["dataset" + j] = str(addresses['labels' + j].replace(path, '') + '.gz').replace('.gz.gz', '.gz')
        config[section]["labels" + j] = str(addresses['labels' + j].replace(path, '') + '.gz').replace('.gz.gz', '.gz')

    for i in ['mordred', 'rdkit', 'maccs', 'external']:
        if i in descriptors and addresses[i + j]:
            try:
                config[section][i + j] = str(addresses[i + j].replace(path, '') + '.gz').replace('.gz.gz', '.gz')
            except:
                config.read(options.data_config)
                config[section][i + j] = str(addresses[i + j].replace(path, '') + '.gz').replace('.gz.gz', '.gz')

    for i in ['morgan', 'spectrophore']:
        if i in descriptors and addresses[i + j]:
            try:
                config[section][i + '_' + str(options.n_bits) + j] = str(addresses[i + j].replace(path, '') + '.gz').replace('.gz.gz', '.gz')
            except:
                config.read(options.data_config)
                config[section][i + '_' + str(options.n_bits) + j] = str(addresses[i + j].replace(path, '') + '.gz').replace('.gz.gz', '.gz')

    with open(options.data_config, 'w') as configfile:
        config.write(configfile)


def drop_nan(x, y, axis=1):
    """
    Remove rows with NaN in data and labes relevant to this rows.
    """
    targ = x.shape[1]
    table = np.c_[x, y]
    table = pd.DataFrame(table)
    table = table.dropna(axis=axis, how='any')
    table = np.array(table)
    x = table[:, 0:targ]
    y = table[:, targ:]
    return x, y


def m_mean(a, column):
    """
    for feature importance
    """
    mean = np.mean(a.T[column])
    for i in range(len(a)):
        a[i][column] = mean
    return a


def clean_data(x, mode="zero"):
    x = np.array(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if isinstance(x[i][j], (str, np.str)):
                try:
                    x[i][j] = float(x[i][j])
                except:
                    x[i][j] = np.nan

    if mode == "mean":
        x = pd.DataFrame(x)
        x = x.apply(lambda l: l.fillna(l.mean()), axis=0)
        x = np.array(x)
    if mode == "zero":
        x = pd.DataFrame(x)
        x = x.fillna(0)
        x = np.array(x)

    for j, col in enumerate(x):
        for i, row in enumerate(col):
            if not isinstance(row, (int, float, np.int64, np.float64)):
                x[j][i] = 0
            if np.isnan(x[j][i]):  # do NOT remove
                if mode == "clean":
                    x[j][i] = 0
            if not np.isfinite(x[j][i]):  # do NOT remove
                x[j][i] = 0

    return x


def split_test_val(split_type, data, split_size, random_state):
    split_size = 1 - split_size * 2
    data = split(split_type, data, split_size, random_state, 'test')

    data["smiles_test"] = np.array(data["smiles_test"])
    data["x_test"] = np.array(data["x_test"])

    split_size = 0.5
    data = split(split_type, data, split_size, random_state, 'val')
    return data


def split(split_type, data, split_size, random_state, name):
    if split_type == "scaffold":
        train, test = scaffold_split(data["smiles_train"], frac_train=1 - split_size)
    elif split_type == "cluster":
        train, test = cluster_split(data["smiles_train"], test_cluster_id=1, n_splits=2)
    elif split_type == "random":
        shuffle = True
        stratify = None
        idx = [i for i in range(len(data["smiles_train"]))]
        train, test = train_test_split(idx, test_size=split_size, stratify=stratify,
                                       shuffle=shuffle, random_state=random_state)
    elif split_type == "stratified":
        shuffle = True
        stratify = data["y_train"]
        idx = [i for i in range(len(data["smiles_train"]))]
        train, test = train_test_split(idx, test_size=split_size, stratify=stratify,
                                       shuffle=shuffle, random_state=random_state)
    else:
        print("Wrong split type")
        sys.exit(0)

    data["x_" + name] = data["x_train"][test]
    data["x_train"] = data["x_train"][train]

    data["y_" + name] = data["y_train"][test]
    data["y_train"] = data["y_train"][train]

    data["smiles_" + name] = data["smiles_train"][test]
    data["smiles_train"] = data["smiles_train"][train]

    data["labels_" + name] = data["labels_train"].iloc[test]
    data["labels_train"] = data["labels_train"].iloc[train]

    data["full_smiles_" + name] = data["full_smiles_train"].iloc[test]
    data["full_smiles_train"] = data["full_smiles_train"].iloc[train]
    return data


#if __name__ == "__main__":
#    """
#    Process dataset.
#    """
#    logger = logging.getLogger(__name__)
#    logger.setLevel(logging.INFO)
#    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
#    data_config = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/data/data_configs/bace.ini"
#    fingerprint = 'morgan'
#    descriptor = 'mordred'
#    n_bits = 256
#    set_targets = [0]
#    split = 0.2
#    random_state = 13
#    verbose = 10
#    get_data(logger, data_config, fingerprint, n_bits, set_targets, random_state, split, verbose, descriptor, False)
