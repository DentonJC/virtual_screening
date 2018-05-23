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


def read_data_config(config_path, descriptors, n_bits, split_type=False): 
    # WARNING: If the value is not in the specified section, but it is in the defaul, then it will be taken from the defaul. It's stupid. 
    data_config = ConfigParser.ConfigParser()

    (dataset_test, dataset_val, labels_train, labels_test, labels_val, maccs_train, maccs_test, maccs_val, morgan_train, morgan_test, morgan_val, spectrophore_train, 
    spectrophore_test, spectrophore_val, mordred_train, mordred_test, mordred_val, rdkit_train, rdkit_test, rdkit_val) = (False, False, False, False, False, False, 
    False, False, False, False, False, False, False, False, False, False, False, False, False, False)

    data_config.read(config_path)

    if split_type is False:
        split_type = 'DEFAULT'

    if split_type not in data_config.sections():
        split_type = 'DEFAULT'

    dataset_train = data_config.get(split_type, 'dataset_train')
    
    # rename dataset if without _train
    if "_train" not in dataset_train:
        path = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
        os.rename(path + dataset_train, path + dataset_train.replace(".csv", "_train.csv"))
        dataset_train = dataset_train.replace(".csv", "_train.csv")

    if data_config.has_option(split_type, 'dataset_test'): dataset_test = data_config.get(split_type, 'dataset_test')
    if data_config.has_option(split_type, 'dataset_val'): dataset_val = data_config.get(split_type, 'dataset_val')
    if data_config.has_option(split_type, 'labels_train'): 
        if data_config.get(split_type, 'labels_train') != data_config.get('DEFAULT', 'labels_train'): # and split_type != 'DEFAULT':
            labels_train = data_config.get(split_type, 'labels_train')
    if data_config.has_option(split_type, 'labels_test'): labels_test = data_config.get(split_type, 'labels_test')
    if data_config.has_option(split_type, 'labels_val'): labels_val = data_config.get(split_type, 'labels_val')
    if data_config.has_option(split_type, 'maccs_train'): 
        if data_config.get(split_type, 'maccs_train') != data_config.get('DEFAULT', 'maccs_train'):
            maccs_train = data_config.get(split_type, 'maccs_train')
    
    if data_config.has_option(split_type, 'maccs_test'): maccs_test = data_config.get(split_type, 'maccs_test')
    if data_config.has_option(split_type, 'maccs_val'): maccs_val = data_config.get(split_type, 'maccs_val')
    if data_config.has_option(split_type, 'morgan_' + str(n_bits) + '_train'): 
        if data_config.has_option('DEFAULT', 'morgan_' + str(n_bits) + '_train'):
            if data_config.get(split_type, 'morgan_' + str(n_bits) + '_train') != data_config.get('DEFAULT', 'morgan_' + str(n_bits) + '_train'):
                morgan_train = data_config.get(split_type, 'morgan_' + str(n_bits) + '_train')
    
    if data_config.has_option(split_type, 'morgan_' + str(n_bits) + '_test'): morgan_test = data_config.get(split_type, 'morgan_' + str(n_bits) + '_test')
    if data_config.has_option(split_type, 'morgan_' + str(n_bits) + '_val'): morgan_val = data_config.get(split_type, 'morgan_' + str(n_bits) + '_val')
    if data_config.has_option(split_type, 'spectrophore_train'): 
         if data_config.has_option('DEFAULT', 'spectrophore_train'):
            if data_config.get(split_type, 'spectrophore_train') != data_config.get('DEFAULT', 'spectrophore_train'):
                spectrophore_train = data_config.get(split_type, 'spectrophore_' + str(n_bits) + '_train')
        
    if data_config.has_option(split_type, 'spectrophore_test'): spectrophore_test = data_config.get(split_type, 'spectrophore_' + str(n_bits) + '_test')
    if data_config.has_option(split_type, 'spectrophore_val'): spectrophore_val = data_config.get(split_type, 'spectrophore_' + str(n_bits) + '_val')
    if data_config.has_option(split_type, 'mordred_train'):
        if data_config.get(split_type, 'mordred_train') != data_config.get('DEFAULT', 'mordred_train'):
            mordred_train = data_config.get(split_type, 'mordred_train')
        
    if data_config.has_option(split_type, 'mordred_test'): mordred_test = data_config.get(split_type, 'mordred_test')
    if data_config.has_option(split_type, 'mordred_val'): mordred_val = data_config.get(split_type, 'mordred_val')
    if data_config.has_option(split_type, 'rdkit_train'):
        if data_config.get(split_type, 'rdkit_train') != data_config.get('DEFAULT', 'rdkit_train'):
            rdkit_train = data_config.get(split_type, 'rdkit_train')
        
    if data_config.has_option(split_type, 'rdkit_test'): rdkit_test = data_config.get(split_type, 'rdkit_test')
    if data_config.has_option(split_type, 'rdkit_val'): rdkit_val = data_config.get(split_type, 'rdkit_val')

    return dataset_train, dataset_test, dataset_val, labels_train, labels_test, labels_val, maccs_train, maccs_test, maccs_val, morgan_train, morgan_test, morgan_val, spectrophore_train, spectrophore_test, spectrophore_val, mordred_train, mordred_test, mordred_val, rdkit_train, rdkit_test, rdkit_val


def cv_splits_load(split_type, data_config, targets):
    loaded_cv = False
    cv_config = ConfigParser.ConfigParser()
    cv_config.read(data_config)
    if split_type is False:
        split_type = 'DEFAULT'

    if split_type not in cv_config.sections():
        split_type = 'DEFAULT'
    
    if cv_config.has_option(split_type, 'cv_' + str(targets)): loaded_cv = cv_config.get(split_type, 'cv_' + str(targets))  
    return loaded_cv
    
    
def cv_splits_save(split_type, n_cv, data_config, targets):
    sv_config = ConfigParser.ConfigParser()

    sv_config.read(data_config)
    try:
        sv_config[split_type]['cv_' + str(targets)] = str(n_cv)
    except:
        with open(data_config, "a") as ini:
            ini.write('[' + split_type + ']')
        sv_config.read(data_config)
        sv_config[split_type]['cv_' + str(targets)] = str(n_cv)
    with open(data_config, 'w') as configfile:
        sv_config.write(configfile)
