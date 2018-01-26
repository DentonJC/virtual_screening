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
from src.report_formatter import create_report, plot_auc

if sys.version_info[0] == 2:
    import ConfigParser
else:
    import configparser


# evaluate
# get_latest_file
# read_data_config
def read_model_config(config_path, section):
    if sys.version_info[0] == 2:
        model_config = ConfigParser.ConfigParser()
    else:
        model_config = configparser.ConfigParser()

    model_config = configparser.ConfigParser()
    model_config.read(config_path)
    epochs = eval(model_config.get('DEFAULT', 'epochs'))   
    rparams = eval(model_config.get(section, 'rparams'))
    gparams = eval(model_config.get(section, 'gparams'))
    return epochs, rparams, gparams


def read_data_config(config_path, section):
    if sys.version_info[0] == 2:
        data_config = ConfigParser.ConfigParser()
    else:
        data_config = configparser.ConfigParser()

    data_config.read(config_path)

    dataset_train = data_config.get('DEFAULT', 'dataset_train')
    
    if data_config.has_option('DEFAULT', 'dataset_test'):
        dataset_test = data_config.get('DEFAULT', 'dataset_test')
    else:
        dataset_test = False
        
    if data_config.has_option('DEFAULT', 'dataset_val'):
        dataset_val = data_config.get('DEFAULT', 'dataset_val')
    else:
        dataset_val = False

    if data_config.has_option('DEFAULT', 'labels_train'):
        labels_train = data_config.get('DEFAULT', 'labels_train')
    else:
        labels_train = False
        
    if data_config.has_option('DEFAULT', 'labels_test'):
        labels_test = data_config.get('DEFAULT', 'labels_test')
    else:
        labels_test = False
        
    if data_config.has_option('DEFAULT', 'labels_val'):
        labels_val = data_config.get('DEFAULT', 'labels_val')
    else:
        labels_val = False
        
    if data_config.has_option('DEFAULT', 'physical_train'):
        physical_train = data_config.get('DEFAULT', 'physical_train')
    else:
        physical_train = False
        
    if data_config.has_option('DEFAULT', 'physical_test'):
        physical_test = data_config.get('DEFAULT', 'physical_test')
    else:
        physical_test = False
        
    if data_config.has_option('DEFAULT', 'physical_val'):
        physical_val = data_config.get('DEFAULT', 'physical_val')
    else:
        physical_val = False
        

    if data_config.has_option(section, 'fingerprint_train'):
        fingerprint_train = data_config.get(section, 'fingerprint_train')
    else:
        fingerprint_train = False
        
    if data_config.has_option(section, 'fingerprint_test'):
        fingerprint_test = data_config.get(section, 'fingerprint_test')
    else:
        fingerprint_test = False
        
    if data_config.has_option(section, 'fingerprint_val'):
        fingerprint_val = data_config.get(section, 'fingerprint_val')
    else:
        fingerprint_val = False
    
    return dataset_train, dataset_test, dataset_val, labels_train, labels_test, labels_val, physical_train, physical_test, physical_val, fingerprint_train, fingerprint_test, fingerprint_val


def start_log(logger, GRID_SEARCH, fingerprint, nBits, config_path, section):
    logger.info("Script adderss: %s", str(sys.argv[0]))
    logger.info("Fingerprint: %s", str(fingerprint))
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


def evaluate(logger, options, random_state, path, model, x_train, x_test, x_val, y_val, y_train, y_test, time_start, rparams, history, section, features, n_jobs, score):
    try:
        """
        pickle.dump(model, open(path+"model.sav", 'wb'))
        model_json = model.to_json()
        with open(path+"model.json", "w") as json_file:
            json_file.write(model_json)
        try:
            copyfile(get_latest_file(path), path + "best_weights.h5")
        except:
            pass
        
        # print and save model summary
        orig_stdout = sys.stdout
        f = open(path + 'model', 'w')
        sys.stdout = f
        logger.info(model.summary())
        sys.stdout = orig_stdout
        f.close()
        """
        y_pred_test = model.predict(x_test)
        result = y_pred_test
        
        save_labels(result, path + "y_pred_test.csv")

        y_pred_train = model.predict(x_train)       
        y_pred_val = model.predict(x_val)
        
        save_labels(y_pred_val, path + "y_pred_val.csv")
        #save_labels(y_val, path + "y_val.csv")

        accuracy_test = accuracy_score(y_test, result)*100
        accuracy_train = accuracy_score(y_train, y_pred_train)*100
        logger.info("Accuracy test: %.2f%%" % (accuracy_test))
    except:
        try:
            pickle.dump(model, open(path+"model.sav", 'wb'))
        except TypeError:
            logger.info("Can not pickle this model")
        y_pred_test = model.predict(x_test)
        
        try:
            result = [round(value) for value in y_pred_test]
        except TypeError:
            y_pred_test = [item for sublist in y_pred_test for item in sublist]
            result = [round(value) for value in y_pred_test]
        
        save_labels(result, path + "y_pred_test.csv")
        
        try:
            y_pred_train = [round(value) for value in y_pred_train]
        except TypeError:
            y_pred_train = [item for sublist in y_pred_train for item in sublist]
            y_pred_train = [round(value) for value in y_pred_train]

        try:
            y_pred_val = [round(value) for value in y_pred_val]
        except TypeError:
            y_pred_val = [item for sublist in y_pred_val for item in sublist]
            y_pred_val = [round(value) for value in y_pred_val]
        
        save_labels(y_pred_val, path + "y_pred_val.csv")
        #save_labels(y_val, path + "y_val.csv")

        accuracy_test = accuracy_score(y_test, result)*100
        accuracy_train = accuracy_score(y_train, y_pred_train)*100
        logger.info("Accuracy test: %.2f%%" % (accuracy_test))
    
    
    rec = recall_score(y_test, result, average=None)
    try:        
        auc = roc_auc_score(y_test, result)
        auc_val = roc_auc_score(y_val, y_pred_val)
    except ValueError:
        auc = '-'
        auc_val = '-'
    f1 = f1_score(y_test, result, average=None)
    
    # find how long the program was running
    tstop = datetime.now()
    timer = tstop - time_start
    print(timer)
    logger.info(timer)

    # create report, prediction and save script and all current models
    try:
        create_report(logger, path, accuracy_test, accuracy_train, rec, auc, f1, timer, rparams, time_start, history, random_state, options, x_train, y_train, x_test, y_test, x_val, y_val, y_pred_train, y_pred_test, y_pred_val, score)
    except:
        create_report(logger, path, accuracy_test, accuracy_train, rec, auc, f1, timer, rparams, time_start, None, random_state, options, x_train, y_train, x_test, y_test, x_val, y_val, y_pred_train, y_pred_test, y_pred_val, score)

    copyfile(sys.argv[0], path + os.path.basename(sys.argv[0]))
    try:
        copytree('src/models', path + 'models')
    except:  # FileNotFoundError not comp python2
        pass

    path_old = path[:-1]

    try:
        print(path_old)
        print(path)
        path = (path[:-8] + '_' + section +  '_' + features +  '_' + str(round(accuracy_test, 3)) +'/').replace(" ", "_")
        os.rename(path_old, path)
    except TypeError:
        pass
    
    logger.info("Done")
    logger.info("Results path: %s", path)

    return accuracy_test, accuracy_train, rec, auc, auc_val, f1
