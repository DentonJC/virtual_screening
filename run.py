#!/usr/bin/env python

"""
An additional script to run a series of experiments described in table like etc/experiments.csv
 where columns are hyperparameters and rows are experiments.
"""
import os
import sys
import math
import random
import pandas as pd
import numpy as np
from moloi.moloi import experiment
# import tensorflow as tf
import multiprocessing
from joblib import Parallel, delayed 

def isnan(x):
    """ Checking if the variable is NaN and type float. """
    return isinstance(x, float) and math.isnan(x)

def f(i, table, random_state, verbose, logger_flag):
    rparams = False
    command = []
    for c, p in enumerate(params):
        if not isnan(table[p][i]):
            if keys[c] in ["-g", "--dummy"]:
                command.append(keys[c])
            elif keys[c] in ["--n_bits", "--n_cv", "--n_jobs", "-p", "--n_iter"]:
                command.append(keys[c])
                command.append(int(table[p][i]))
            elif keys[c] in ["--split"]:
                command.append(keys[c])
                command.append(float(table[p][i]))
            else:
                command.append(keys[c])
                command.append(str(table[p][i]))
        
    command.append("-e")
    command.append(experiments_file)
    for j in range(int(((table.shape[1] - len(params)) / len(result_cols)))):
        accuracy_test = accuracy_train = rec = auc = f1 = '-'
        final_command = command + ["-t"] + [int(j)]
        print(final_command)
        if isnan(table.iloc[i, j*len(result_cols) + len(params)]):    
            if not common_gridsearch:
                rparams = False
            accuracy_test, accuracy_train, rec, auc, auc_val, f1, rparams, model_address = experiment(final_command, random_state, rparams, verbose, logger_flag)
            accuracy_test, accuracy_train, rec, auc, auc_val, f1 = round(accuracy_test, 4), round(accuracy_train, 4), (round(rec[0], 4), round(rec[1], 4)), round(auc, 4), round(auc_val, 4), (round(f1[0], 4), round(f1[1], 4))
            balanced_accuracy = (rec[0] + rec[1]) / 2
            gparams = str(rparams)
            table = pd.read_csv(experiments_file)

            for p, r in enumerate(result_cols):
                table.iloc[i, j * len(result_cols) + len(params) + p] = eval(r)

            if model_address:
                table.iloc[i, 5] = str(model_address) # set on Load model column
                if "--load_model" not in command and common_gridsearch:
                    command.append("--load_model")
                    command.append(model_address)
                    
            table.to_csv(experiments_file, index=False)
            logger_flag = True

def main(experiments_file, common_gridsearch, random_state, result_cols, keys, params, verbose, n_jobs):
    """
    Check the rows of experiments_file in a loop. If there are no results in the row (empty fields after len(cols)), 
    it takes all values in this row and calls the experiment function until all result fields are filled with step len(result_cols).
    
    Params
    ------
    params: list
        the names of the columns of not positional arguments in experiments_file
    keys: list
        not positional arguments of experiment function (run_model.py) in the same order as params
    result_cols: list
        the result metrics which will be added to experiments_file [accuracy_test, accuracy_train, rec, auc, auc_val, f1, gparams]
    common_gridsearch: bool
        one gridsearch for all experiments in row
    random_state: int
        random state of all
    experiments_file: string
        path to experiments table
    """
    logger_flag = False
    if not random_state and not isinstance(random_state, int): random_state = random.randint(1, 100)
    np.random.seed(random_state)
    # tf.set_random_seed(random_state)
    table = pd.read_csv(experiments_file)
    if n_jobs > table.shape[0]:
        n_jobs = table.shape[0] - 1
    Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(f)(i, table, random_state, verbose, logger_flag) for (i) in range(table.shape[0]))


if __name__ == "__main__":
    keys = ["--load_model", "--output", "--model_config", "--descriptors", "--n_bits", "--n_cv", 
            "--n_jobs", "-p", "-g", "--n_iter", "--metric", "--split_type", "--split_s", '--select_model', '--data_config', '--section']
    params = ["Load model", "Output", "Model config", "Descriptors", "n_bits", "n_cv", "n_jobs", "Patience", 
            "Gridsearch", "n_iter", "Metric", "Split type", "Split size", 'Model', 'Data config', 'Section']
    result_cols = ['balanced_accuracy', 'auc', 'auc_val', 'gparams']
    common_gridsearch = True
    random_state = 1337
    experiments_file = 'etc/experiments_clintox.csv'
    verbose = 10
    n_jobs=multiprocessing.cpu_count()
    main(experiments_file, common_gridsearch, random_state, result_cols, keys, params, verbose, n_jobs)
