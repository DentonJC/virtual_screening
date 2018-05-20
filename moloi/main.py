#!/usr/bin/env python

import os
import sys
import csv
import glob
import math
import pickle
import getopt
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from shutil import copyfile, copytree
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef, make_scorer
from moloi.report_formatter import create_report, plot_auc, plot_TSNE

if sys.version_info[0] == 2:
    import ConfigParser
else:
    import configparser as ConfigParser


# evaluate
# get_latest_file
# read_data_config
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


def cv_splits_load(split_type, data_config):
    loaded_cv = False
    cv_config = ConfigParser.ConfigParser()
    cv_config.read(data_config)
    if split_type is False:
        split_type = 'DEFAULT'

    if split_type not in cv_config.sections():
        split_type = 'DEFAULT'
    
    if cv_config.has_option(split_type, 'cv'): loaded_cv = cv_config.get(split_type, 'cv')   
    return loaded_cv
    
    
def cv_splits_save(split_type, n_cv, data_config):
    sv_config = ConfigParser.ConfigParser()

    sv_config.read(data_config)
    try:
        sv_config[split_type]["cv"] = str(n_cv)
    except:
        with open(data_config, "a") as ini:
            ini.write('[' + split_type + ']')
        sv_config.read(data_config)
        sv_config[split_type]["cv"] = str(n_cv)
    with open(data_config, 'w') as configfile:
        sv_config.write(configfile)


def start_log(logger, GRID_SEARCH, nBits, config_path, section, descriptors):
    logger.info("Script adderss: %s", str(sys.argv[0]))
    logger.info("Descriptors: %s", str(descriptors))
    logger.info("n_bits: %s", str(nBits))
    logger.info("Config file: %s", str(config_path))
    logger.info("Section: %s", str(section))
    if GRID_SEARCH:
        logger.info("Grid search")


def get_latest_file(path):
    """
    Return the path to the last (and best) checkpoint.
    """
    list_of_files = glob.iglob(path + "results/*")
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    _, filename = os.path.split(latest_file)

    return path+"results/"+filename


def save_labels(arr, filename):
    pd_array = pd.DataFrame(arr)
    pd_array.index.names = ["Id"]
    pd_array.columns = ["Prediction"]
    pd_array.to_csv(filename)


def make_scoring(metric):
    scoring = make_scorer(accuracy_score)
    if metric in 'accuracy':
        scoring = make_scorer(accuracy_score)
    if metric in 'roc_auc':
        scoring = make_scorer(roc_auc_score)
    if metric in 'f1':
        scoring = make_scorer(f1_score)
    if metric in 'matthews':
        scoring = make_scorer(matthews_corrcoef)
    return scoring


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


def evaluate(logger, options, random_state, path, model, x_train, x_test, x_val, y_val, y_train, y_test, time_start, rparams, history, section, n_jobs, descriptors):
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    save_labels(y_pred_train, path + "y_pred_test.csv")
    y_pred_val = model.predict(x_val)
    save_labels(y_pred_val, path + "y_pred_val.csv")
    
    y_pred_test = np.ravel(y_pred_test)
    y_pred_train = np.ravel(y_pred_train)
    y_pred_val = np.ravel(y_pred_val)

    try:
        y_pred_test = [int(round(value)) for value in y_pred_test]
        y_pred_train = [int(round(value)) for value in y_pred_train]
    except ValueError:
        logger.error("Model is not trained")
        sys.exit(0)
        
    accuracy_test = accuracy_score(list(np.ravel(y_test)), y_pred_test)*100
    accuracy_train = accuracy_score(list(np.ravel(y_train)), y_pred_train)*100
    logger.info("Accuracy test: %.2f%%" % (accuracy_test))
        
    rec = recall_score(y_test, y_pred_test, average=None)
    
    try:
        train_proba = model.predict_proba(x_train)
        test_proba = model.predict_proba(x_test)
        val_proba = model.predict_proba(x_val)
        try:
            train_proba = train_proba[:,1]
            val_proba = val_proba[:,1]
            test_proba = test_proba[:,1]
        except:
            pass
        auc_train = roc_auc_score(y_train, train_proba)
        auc_test = roc_auc_score(y_test, test_proba)
        auc_val = roc_auc_score(y_val, val_proba)
    except ValueError:
        auc_train = '-'
        auc_test = '-'
        auc_val = '-'

    f1 = f1_score(y_test, y_pred_test, average=None)
    try: # is model after gridsearch replace by reading from options.output + "gridsearch.csv"
        score = pd.DataFrame(model.cv_results_)
    except:
        score = False
    """
    # balanced acc
    num_positive = float(np.count_nonzero(y_test))
    num_negative = float(len(y_test) - num_positive)
    pos_weight = num_negative / num_positive
    weights = np.ones_like(y_test)
    weights[y_train != 0] = pos_weight
    b_acc = accuracy_score(y_test, y_pred_test, sample_weight=weights)
    """
    
    # find how long the program was running
    tstop = datetime.now()
    timer = tstop - time_start
    logger.info(timer)
    # create report, prediction and save script and all current models
    create_report(logger, path, accuracy_test, accuracy_train, rec, auc_train, auc_test, auc_val, train_proba, test_proba, val_proba, f1, timer, rparams, time_start, history, random_state, options, x_train, y_train, x_test, y_test, x_val, y_val, y_pred_train, y_pred_test, y_pred_val, score)
    
    copyfile(sys.argv[0], path + os.path.basename(sys.argv[0]))
    try:
        copytree('moloi/models', path + 'models')
    except:  # FileNotFoundError not comp python2
        pass

    path_old = path[:-1]

    try:
        path = (path[:-8] + '_' + section +  '_' + str(descriptors) +  '_' + str(round(accuracy_test, 3)) +'/').replace(" ", "_")
        os.rename(path_old, path)
    except TypeError:
        pass
    
    #plot_TSNE(x_train, y_train, path)
    logger.info("Results path: %s", path)

    return accuracy_test, accuracy_train, rec, auc_test, auc_val, f1, path
