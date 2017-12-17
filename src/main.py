#!/usr/bin/env python

import os
import sys
import csv
import glob
import math
import getopt
import logging
import configparser
import numpy as np
import pandas as pd
from datetime import datetime
from shutil import copyfile, copytree
from sklearn.metrics import accuracy_score, recall_score
from keras.optimizers import Adam, Nadam, Adamax, RMSprop, Adagrad, Adadelta, SGD
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from src.report_formatter import create_report
config = configparser.ConfigParser()


def create_callbacks(output, patience, data):
    if not os.path.exists(output):
        os.makedirs(output)
    if not os.path.exists(output+"results/*"):
        os.makedirs(output+"results/")
            
    filepath = output + "results/" + "weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience, verbose=0, mode='auto')
    csv_logger = CSVLogger(output + 'history_' + os.path.basename(sys.argv[0]).replace(".py", "") +
                           "_" + os.path.basename(data).replace(".csv", "") + '.csv', append=True, separator=';')
    callbacks_list = [checkpoint, stopping, csv_logger]
    return callbacks_list


def read_config(config_path, section):
    config.read(config_path)
    def_config = config['DEFAULT']
    n_folds = eval(def_config['n_folds'])
    epochs = eval(def_config['epochs'])
    model_config = config[section]
    rparams = eval(model_config['rparams'])
    gparams = eval(model_config['gparams'])
    return n_folds, epochs, rparams, gparams


def start_log(logger, DUMMY, GRID_SEARCH, fingerprint, nBits, config_path, filename, section):
    logger.info("Script adderss: %s", str(sys.argv[0]))
    logger.info("Data file: %s", str(filename))
    logger.info("Fingerprint: %s", str(fingerprint))
    logger.info("n_bits: %s", str(nBits))
    logger.info("Config file: %s", str(config_path))
    logger.info("Section: %s", str(section))
    if DUMMY:
        logger.info("Dummy")
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


def load_model(loaded_model_json):
    json_file = open(loaded_model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model


class Logger(object):
    """https://stackoverflow.com/questions/11325019/output-on-the-console-and-file-using-python"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def compile_optimizer(optimizer, learning_rate=0.1, momentum=0.1):
    if optimizer == 'Adam':
        return Adam(lr=learning_rate)
    elif optimizer == 'Nadam':
        return Nadam(lr=learning_rate)
    elif optimizer == 'Adamax':
        return Adamax(lr=learning_rate)
    elif optimizer == 'RMSprop':
        return RMSprop(lr=learning_rate)
    elif optimizer == 'Adagrad':
        return Adagrad(lr=learning_rate)
    elif optimizer == 'Adadelta':
        return Adadelta(lr=learning_rate)
    else:
        return SGD(lr=learning_rate, momentum=momentum)


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
    

def isnan(x):
    return isinstance(x, float) and math.isnan(x)
   

def evaluate(logger, options, random_state, path, model, x_train, x_test, x_val, y_train, y_test, y_val, time_start, rparams, history, data, section, features, n_jobs):
    try:
        model_json = model.to_json()
        with open(path+"model.json", "w") as json_file:
            json_file.write(model_json)
        if rparams.get("metrics") == ['accuracy']:
            copyfile(get_latest_file(path), path + "best_weights.h5")

        # print and save model summary
        orig_stdout = sys.stdout
        f = open(path + 'model', 'w')
        sys.stdout = f
        logger.info(model.summary())
        sys.stdout = orig_stdout
        f.close()

        # evaluate
        score = model.evaluate(x_test, y_test, batch_size=rparams.get("batch_size", 32), verbose=1)
        logger.info('Score: %1.3f' % score[0])
        logger.info('Accuracy: %1.3f' % score[1])
        logger.info("PREDICT")
        y_pred = model.predict(x_test)
        result = [np.argmax(i) for i in y_pred]
        y_pred_train = model.predict(x_train)
        result_train = [np.argmax(i) for i in y_pred_train]
        
        accuracy = accuracy_score(y_test, result)*100
        accuracy_train = accuracy_score(y_train, result_train)*100
        
        save_labels(result, path + "y_pred.csv")
    except:
        y_pred_test = model.predict(x_test)
        result = [round(value) for value in y_pred_test]
        save_labels(result, path + "y_pred.csv")

        y_pred_train = model.predict(x_train)
        p_train = [round(value) for value in y_pred_train]

        accuracy = accuracy_score(y_test, result)*100
        accuracy_train = accuracy_score(y_train, p_train)*100
        logger.info("Accuracy test: %.2f%%" % (accuracy))
        logger.info("Accuracy train: %.2f%%" % (accuracy_train))
        
        score = [1-accuracy_score(y_pred_test, y_test), accuracy_score(y_pred_test, y_test)]
    
    rec = recall_score(y_test, result, average=None)
    
    # find how long the program was running
    tstop = datetime.now()
    timer = tstop - time_start
    print(timer)
    logger.info(timer)

    # create report, prediction and save script and all current models
    try:
        create_report(path, score, timer, rparams, time_start, history, random_state, options)
    except:
        create_report(path, score, timer, rparams, tstart, None, random_state, options)
    copyfile(sys.argv[0], path + os.path.basename(sys.argv[0]))
    copytree('src/models', path + 'models')
    path_old = path
    path = (path[:-1] + ' ' + data +  ' ' + section +  ' ' + features +  ' ' + str(round(score[1], 3)) +'/').replace(".py", "").replace(".csv", "").replace("src/","").replace("data/preprocessed/","").replace('data/','') # ugly! remove address of file
    os.rename(path_old,path)
    
    logger.info("Done")
    logger.info("Results path: %s", path)
    return accuracy, accuracy_train, rec
