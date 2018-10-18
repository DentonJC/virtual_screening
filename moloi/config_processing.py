import os
import sys
from moloi.dictionaries import addresses_dict

if sys.version_info[0] == 2:
    import ConfigParser
else:
    import configparser as ConfigParser


def read_model_config(config_path, section):
    model_config = ConfigParser.ConfigParser()

    model_config.read(config_path)
    epochs = eval(model_config.get('DEFAULT', 'epochs'))
    rparams = eval(model_config.get(section, 'rparams'))
    gparams = eval(model_config.get(section, 'gparams'))
    return epochs, rparams, gparams


def read_data_config(config_path, n_bits, split_type=False, split_size=0.1):
    # WARNING: If the value is not in the specified section, but it is in the defaul,
    # then it will be taken from the defaul. It's stupid.
    data_config = ConfigParser.ConfigParser()
    addresses = addresses_dict()
    data_config.read(config_path)

    if split_type is False:
        section = 'init'
    else:
        section = split_type + " " + str(split_size)

    if split_type + " " + str(split_size) not in data_config.sections():
        section = 'init'
    else:
        section = split_type + " " + str(split_size)

    ################################################
    # WARNING! Disable loading of splitted datasets#
    ################################################
    section = 'init'
    ################################################
    for j in ['_train', '_test', '_val']:
        if data_config.has_option(section, 'dataset' + j):
            addresses['dataset' + j] = data_config.get(section, 'dataset' + j)

        if addresses['dataset' + j]:
            if j not in addresses['dataset' + j]: # Rename "dataset.csv" to "dataset_train.csv" and change address
                path = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
                os.rename(path + addresses['dataset' + j],
                          path + addresses['dataset' + j].replace(".csv", "_train.csv"))
                addresses['dataset' + j] = addresses['dataset' + j].replace(".csv", "_train.csv")

        for i in ['labels', 'maccs', 'mordred', 'rdkit', 'external']:
            if data_config.has_option(section, i + j):
                addresses[i + j] = data_config.get(section, i + j)

        for i in ['morgan_', 'spectrophore_']:
            if data_config.has_option(section, i + str(n_bits) + j):
                addresses[i + j] = data_config.get(section, i + str(n_bits) + j)

    return addresses


def cv_splits_load(split_type, split_size, data_config, targets):
    loaded_cv = False
    cv_config = ConfigParser.ConfigParser()
    cv_config.read(data_config)
    if split_type is False:
        section = 'init'
    else:
        section = split_type + " " + str(split_size)

    if split_type + " " + str(split_size) not in cv_config.sections():
        section = 'init'
    else:
        section = split_type + " " + str(split_size)

    if cv_config.has_option(section, 'cv_' + str(targets)):
        loaded_cv = cv_config.get(section, 'cv_' + str(targets))
    return loaded_cv


def cv_splits_save(split_type, split_size, n_cv, data_config, targets):
    sv_config = ConfigParser.ConfigParser()

    sv_config.read(data_config)
    try:
        sv_config[split_type + " " + str(split_size)]['cv_' + str(targets)] = str(n_cv)
    except:
        with open(data_config, "a") as ini:
            ini.write('[' + split_type + " " + str(split_size) + ']')
        sv_config.read(data_config)
        sv_config[split_type + " " + str(split_size)]['cv_' + str(targets)] = str(n_cv)
    with open(data_config, 'w') as configfile:
        sv_config.write(configfile)
