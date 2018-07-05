import os
import sys

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


def read_data_config(config_path, descriptors, n_bits, split_type=False, split_size=0.1): 
    # WARNING: If the value is not in the specified section, but it is in the defaul, then it will be taken from the defaul. It's stupid. 
    data_config = ConfigParser.ConfigParser()

    (dataset_test, dataset_val, labels_train, labels_test, labels_val, maccs_train, maccs_test, maccs_val, morgan_train, morgan_test, morgan_val, spectrophore_train, 
    spectrophore_test, spectrophore_val, mordred_train, mordred_test, mordred_val, rdkit_train, rdkit_test, rdkit_val) = (False, False, False, False, False, False, 
    False, False, False, False, False, False, False, False, False, False, False, False, False, False)
    external_train, external_test, external_val  = False, False, False

    data_config.read(config_path)

    if split_type is False:
        section = 'init'
    else:
        section =  split_type + " " + str(split_size)

    if split_type + " " + str(split_size) not in data_config.sections():
        #dataset_train = data_config.get('init', 'dataset_train')
        #return dataset_train, dataset_test, dataset_val, labels_train, labels_test, labels_val, maccs_train, maccs_test, maccs_val, morgan_train, morgan_test, morgan_val, spectrophore_train, spectrophore_test, spectrophore_val, mordred_train, mordred_test, mordred_val, rdkit_train, rdkit_test, rdkit_val, external_train, external_test, external_val

        section = 'init'
    else:
        section =  split_type + " " + str(split_size)
    
    ################################################
    # WARNING! Disable loading of splitted datasets#
    ################################################
    section = 'init'
    ################################################

    dataset_train = data_config.get(section, 'dataset_train')
    
    # rename dataset if without _train
    if "_train" not in dataset_train:
        path = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
        os.rename(path + dataset_train, path + dataset_train.replace(".csv", "_train.csv"))
        dataset_train = dataset_train.replace(".csv", "_train.csv")

    if data_config.has_option(section, 'dataset_test'): dataset_test = data_config.get(section, 'dataset_test')
    if data_config.has_option(section, 'dataset_val'): dataset_val = data_config.get(section, 'dataset_val')
    
    if dataset_test:
        if "_test" not in dataset_test:
            path = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
            os.rename(path + dataset_test, path + dataset_test.replace(".csv", "_test.csv"))
            dataset_test = dataset_test.replace(".csv", "_test.csv")
    
    if dataset_val:
        if "_val" not in dataset_val:
            path = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
            os.rename(path + dataset_val, path + dataset_val.replace(".csv", "_val.csv"))
            dataset_val = dataset_val.replace(".csv", "_val.csv")
    
    if data_config.has_option(section, 'labels_train'): 
        #if data_config.get(section, 'labels_train') != data_config.get('DEFAULT', 'labels_train'): # and split_type != 'DEFAULT':
        labels_train = data_config.get(section, 'labels_train')
    
    if data_config.has_option(section, 'external_train'): 
        external_train = data_config.get(section, 'external_train')
    if data_config.has_option(section, 'external_test'): 
        external_test = data_config.get(section, 'external_test')
    if data_config.has_option(section, 'external_val'): 
        external_val = data_config.get(section, 'external_val')

    if data_config.has_option(section, 'labels_test'): labels_test = data_config.get(section, 'labels_test')
    if data_config.has_option(section, 'labels_val'): labels_val = data_config.get(section, 'labels_val')
    if data_config.has_option(section, 'maccs_train'): 
        #if data_config.get(section, 'maccs_train') != data_config.get('DEFAULT', 'maccs_train'):
        maccs_train = data_config.get(section, 'maccs_train')
    
    if data_config.has_option(section, 'maccs_test'): maccs_test = data_config.get(section, 'maccs_test')
    if data_config.has_option(section, 'maccs_val'): maccs_val = data_config.get(section, 'maccs_val')
    if data_config.has_option(section, 'morgan_' + str(n_bits) + '_train'): 
        #if data_config.has_option('DEFAULT', 'morgan_' + str(n_bits) + '_train'):
        #if data_config.get(section, 'morgan_' + str(n_bits) + '_train') != data_config.get('DEFAULT', 'morgan_' + str(n_bits) + '_train'):
        morgan_train = data_config.get(section, 'morgan_' + str(n_bits) + '_train')
    
    if data_config.has_option(section, 'morgan_' + str(n_bits) + '_test'): morgan_test = data_config.get(section, 'morgan_' + str(n_bits) + '_test')
    if data_config.has_option(section, 'morgan_' + str(n_bits) + '_val'): morgan_val = data_config.get(section, 'morgan_' + str(n_bits) + '_val')
    if data_config.has_option(section, 'spectrophore_train'): 
        #if data_config.has_option('DEFAULT', 'spectrophore_train'):
        #if data_config.get(section, 'spectrophore_train') != data_config.get('DEFAULT', 'spectrophore_train'):
        spectrophore_train = data_config.get(section, 'spectrophore_' + str(n_bits) + '_train')
        
    if data_config.has_option(section, 'spectrophore_test'): spectrophore_test = data_config.get(section, 'spectrophore_' + str(n_bits) + '_test')
    if data_config.has_option(section, 'spectrophore_val'): spectrophore_val = data_config.get(section, 'spectrophore_' + str(n_bits) + '_val')
    if data_config.has_option(section, 'mordred_train'):
        #if data_config.get(section, 'mordred_train') != data_config.get('DEFAULT', 'mordred_train'):
        mordred_train = data_config.get(section, 'mordred_train')
        
    if data_config.has_option(section, 'mordred_test'): mordred_test = data_config.get(section, 'mordred_test')
    if data_config.has_option(section, 'mordred_val'): mordred_val = data_config.get(section, 'mordred_val')
    if data_config.has_option(section, 'rdkit_train'):
        #if data_config.get(section, 'rdkit_train') != data_config.get('DEFAULT', 'rdkit_train'):
        rdkit_train = data_config.get(section, 'rdkit_train')
        
    if data_config.has_option(section, 'rdkit_test'): rdkit_test = data_config.get(section, 'rdkit_test')
    if data_config.has_option(section, 'rdkit_val'): rdkit_val = data_config.get(section, 'rdkit_val')

    return dataset_train, dataset_test, dataset_val, labels_train, labels_test, labels_val, maccs_train, maccs_test, maccs_val, morgan_train, morgan_test, morgan_val, spectrophore_train, spectrophore_test, spectrophore_val, mordred_train, mordred_test, mordred_val, rdkit_train, rdkit_test, rdkit_val, external_train, external_test, external_val


def cv_splits_load(split_type, split_size, data_config, targets):
    loaded_cv = False
    cv_config = ConfigParser.ConfigParser()
    cv_config.read(data_config)
    if split_type is False:
        section = 'init'
    else:
        section =  split_type + " " + str(split_size)

    if split_type + " " + str(split_size) not in cv_config.sections():
        section = 'init'
    else:
        section =  split_type + " " + str(split_size)
    
    if cv_config.has_option(section, 'cv_' + str(targets)): loaded_cv = cv_config.get(section, 'cv_' + str(targets))  
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
