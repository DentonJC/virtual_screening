###########
# DISABLED
###########

import sys
from moloi.data_processing import create_addr, save_files, filling_config

if sys.version_info[0] == 2:  # for Python2
    import ConfigParser
else:
    import configparser as ConfigParser


def split_x(x, n_bits, descriptors):
    maccs, morgan, mordred, rdkit, spectrophore, external = False, False, False, False, False, False
    if 'maccs' in descriptors:
        maccs = x[:, :167]
        x = x[:, 167:]

    if 'morgan' in descriptors:
        morgan = x[:, :n_bits]
        x = x[:, n_bits:]

    if 'spectrophore' in descriptors:
        spectrophore = x[:, :n_bits]
        x = x[:, n_bits:]

    if 'mordred' in descriptors:
        mordred = x[:, :len(mordred_fetures_names())]
        x = x[:, len(mordred_fetures_names()):]

    if 'rdkit' in descriptors:
        rdkit = x[:, :len(rdkit_fetures_names())]
        x = x[:, len(rdkit_fetures_names()):]

    if 'external' in descriptors:
        external = x

    return maccs, morgan, mordred, rdkit, spectrophore, external


def save_files(train, test, val, train_addr, test_addr, val_addr):
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    val = pd.DataFrame(val)
    train.to_csv(train_addr, compression="gzip", sep=",", index=False)
    test.to_csv(test_addr, compression="gzip", sep=",", index=False)
    val.to_csv(val_addr, compression="gzip", sep=",", index=False)


def create_addr(path, filename, part, descr, split_type, split_size):
    address = filename+"_" + part + "_" + descr
    address = address.replace("_train_train", "_train").replace("_test_test", "_test").replace("_val_val", "_val")
    head, _sep, tail = address.rpartition('/')
    address = path + "/data/preprocessed/"+descr+"/" + tail + '_' + split_type + '_' + str(split_size) + '.csv.gz'
    return address


def saving_splitted_dataset(path, labels_train, labels_test, labels_val, smiles_train, smiles_test, smiles_val, filename_train, filename_test, filename_val, maccs_train, maccs_test, maccs_val, morgan_train, morgan_test, morgan_val, mordred_train, mordred_test, mordred_val, rdkit_train, rdkit_test, rdkit_val, spectrophore_train, spectrophore_test, spectrophore_val, external_train, external_test, external_val, options):
    labels_train["smiles"] = smiles_train
    labels_test["smiles"] = smiles_test
    labels_val["smiles"] = smiles_val
    names = ["label"]*(labels_train.shape[1]-1) + ["smiles"]

    labels_train.columns = names
    labels_test.columns = names
    labels_val.columns = names

    labels_train_address = create_addr(path, filename_train, "train", "labels", options.split_type, options.split_s)
    labels_test_address = create_addr(path, filename_test, "test", "labels", options.split_type, options.split_s)
    labels_val_address = create_addr(path, filename_val, "val", "labels", options.split_type, options.split_s)
    save_files(labels_train, labels_test, labels_val, labels_train_address, labels_test_address, labels_val_address)

    if maccs_train is not False:
        maccs_train_address = create_addr(path, filename_train, "train", "maccs", options.split_type, options.split_s)
        maccs_test_address = create_addr(path, filename_test, "test", "maccs", options.split_type, options.split_s)
        maccs_val_address = create_addr(path, filename_val, "val", "maccs", options.split_type, options.split_s)
        save_files(maccs_train, maccs_test, maccs_val, maccs_train_address, maccs_test_address, maccs_val_address)
    else:
        maccs_train_address = False
        maccs_test_address = False
        maccs_val_address = False
    if morgan_train is not False:
        morgan_train_address = create_addr(path, filename_train, "train", "morgan", options.split_type, options.split_s)
        morgan_train_address = morgan_train_address.replace("_morgan", "_morgan_" + str(options.n_bits))
        morgan_test_address = create_addr(path, filename_test, "test", "morgan", options.split_type, options.split_s)
        morgan_test_address = morgan_test_address.replace("_morgan", "_morgan_" + str(options.n_bits))
        morgan_val_address = create_addr(path, filename_val, "val", "morgan", options.split_type, options.split_s)
        morgan_val_address = morgan_val_address.replace("_morgan", "_morgan_" + str(options.n_bits))
        save_files(morgan_train, morgan_test, morgan_val, morgan_train_address, morgan_test_address, morgan_val_address)
    else:
        morgan_train_address = False
        morgan_test_address = False
        morgan_val_address = False
    if mordred_train is not False:
        mordred_train_address = create_addr(path, filename_train, "train", "mordred",
                                            options.split_type, options.split_s)
        mordred_test_address = create_addr(path, filename_test, "test", "mordred",
                                           options.split_type, options.split_s)
        mordred_val_address = create_addr(path, filename_val, "val", "mordred", options.split_type, options.split_s)
        save_files(mordred_train, mordred_test, mordred_val, mordred_train_address,
                   mordred_test_address, mordred_val_address)
    else:
        mordred_train_address = False
        mordred_test_address = False
        mordred_val_address = False
    if rdkit_train is not False:
        rdkit_train_address = create_addr(path, filename_train, "train", "rdkit",
                                          options.split_type, options.split_s)
        rdkit_test_address = create_addr(path, filename_test, "test", "rdkit",
                                         options.split_type, options.split_s)
        rdkit_val_address = create_addr(path, filename_val, "val", "rdkit", options.split_type, options.split_s)
        save_files(rdkit_train, rdkit_test, rdkit_val, rdkit_train_address,
                   rdkit_test_address, rdkit_val_address)
    else:
        rdkit_train_address = False
        rdkit_test_address = False
        rdkit_val_address = False
    if spectrophore_train is not False:
        spectrophore_train_address = create_addr(path, filename_train, "train", "spectrophore",
                                                 options.split_type, options.split_size)
        spectrophore_test_address = create_addr(path, filename_test, "test", "spectrophore",
                                                options.split_type, options.split_size)
        spectrophore_val_address = create_addr(path, filename_val, "val", "spectrophore",
                                               options.split_type, options.split_size)
        save_files(spectrophore_train, spectrophore_test, spectrophore_val,
                   spectrophore_train_address, spectrophore_test_address, spectrophore_val_address)
    else:
        spectrophore_train_address = False
        spectrophore_test_address = False
        spectrophore_val_address = False
    if external_train is not False:
        external_val_address = create_addr(path, filename_val, "val", "external",
                                           options.split_type, options.split_size)
        external_train_address = create_addr(path, filename_train, "train", "external",
                                             options.split_type, options.split_size)
        external_test_address = create_addr(path, filename_test, "test", "external",
                                            options.split_type, options.split_size)
        save_files(external_train, external_test, external_val, external_train_address,
                   external_test_address, external_val_address)
    else:
        external_val_address = False
        external_train_address = False
        external_test_address = False

    ini = open(options.data_config, "a")
    ini.write('[' + options.split_type + " " + str(options.split_size) + ']' + '\n')
    ini.write("dataset_train = "+str(labels_train_address.replace(path, ''))+'\n')
    ini.write("dataset_test = "+str(labels_test_address.replace(path, ''))+'\n')
    ini.write("dataset_val = "+str(labels_val_address.replace(path, ''))+'\n')
    ini.close()

    files_config = ConfigParser.ConfigParser()

    files_config.read(options.data_config)
    filling_config(path, options.data_config, "_train", options.descriptors, options.n_bits, options.split_type, options.split_size, labels_train_address, mordred_train_address, rdkit_train_address, maccs_train_address, morgan_train_address, spectrophore_train_address, external_train_address)
    filling_config(path, options.data_config, "_test", options.descriptors, options.n_bits, options.split_type, options.split_size, labels_test_address, mordred_test_address, rdkit_test_address, maccs_test_address, morgan_test_address, spectrophore_test_address, external_test_address)
    filling_config(path, options.data_config, "_val", options.descriptors, options.n_bits, options.split_type, options.split_size, labels_val_address, mordred_val_address, rdkit_val_address, maccs_val_address, morgan_val_address, spectrophore_val_address, external_val_address)
