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
from src.run_model import script


def isnan(x):
    """ Checking if the variable is NaN and type float. """
    return isinstance(x, float) and math.isnan(x)


def main(experiments_file, common_gridsearch, random_state, result_cols, keys, params, pos_params, verbose):
    """
    Check the rows of experiments_file in a loop. If there are no results in the row (empty fields after len(cols)), 
    it takes all values in this row and calls the experiment function until all result fields are filled with step len(result_cols).
    
    Params
    ------
    params: list
        the names of the columns of not positional arguments in experiments_file
    keys: list
        not positional arguments of experiment function (run_model.py) in the same order as params
    pos_params: list
        the names of the columns of positional arguments in experiments_file
    result_cols: list
        the result metrics which will be added to experiments_file [accuracy_test, accuracy_train, rec, auc, auc_val, f1, gparams]
    common_gridsearch: bool
        one gridsearch for all experiments in row
    random_state: int
        random state of all
    experiments_file: string
        path to experiments table
    """
    if not random_state and not isinstance(random_state, int): random_state = random.randint(1, 100)

    table = pd.read_csv(experiments_file)

    for i in range(table.shape[0]):
        rparams = False
        command = []
        for p in pos_params:
            command.append(str(table[p][i]))
        
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
        
        for j in range(int(((table.shape[1] - len(params) - len(pos_params)) / len(result_cols)))):
            accuracy_test = accuracy_train = rec = auc = f1 = '-'
            command.append("-t")
            command.append(int(j))
            print(command)
            if isnan(table.iloc[i, j*len(result_cols) + len(params) + len(pos_params)]):    
                if common_gridsearch:
                    rparams = False

                accuracy_test, accuracy_train, rec, auc, auc_val, f1, rparams = script(command, random_state, rparams, verbose)
                accuracy_test, accuracy_train, rec, auc, auc_val, f1 = round(accuracy_test, 4), round(accuracy_train, 4), (round(rec[0], 4), round(rec[1], 4)), round(auc, 4), round(auc_val, 4), (round(f1[0], 4), round(f1[1], 4))
                gparams = str(rparams)
                table = pd.read_csv(experiments_file)
                
                for p, r in enumerate(result_cols):
                    table.iloc[i, j * len(result_cols) + len(params) + len(pos_params) + p] = eval(r)

                table.to_csv(experiments_file, index=False)


if __name__ == "__main__":
    keys = ["--load_model", "--output", "--model_config", "--fingerprint", "--n_bits", "--n_cv", 
            "--n_jobs", "-p", "-g", "--n_iter", "--features", "--metric", "--split"]
    params = ["Load model", "Output", "Model config", "Fingerprint", "n_bits", "n_cv", "n_jobs", "Patience", 
            "Gridsearch", "n_iter", "Features", "Metric", "Split"]
    pos_params = ['Model', 'Data config', 'Section']
    result_cols = ['rec[0]', 'rec[1]', 'auc', 'auc_val', 'gparams']
    common_gridsearch = True
    random_state = 42
    experiments_file = 'etc/test.csv'
    verbose = 10
    main(experiments_file, common_gridsearch, random_state, result_cols, keys, params, pos_params, verbose)
