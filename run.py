#!/usr/bin/env python

"""
An additional script to run a series of experiments described in the .csv file.
"""
import os
import sys
import math
import random
import pandas as pd
from src.run_model import script


def isnan(x):
    """ Checks if the variable is empty (NaN and float). """
    return isinstance(x, float) and math.isnan(x)


def main(experiments_file, common_gridsearch, random_state, n_cols):
    """
    Checks the rows of experiments_file in a loop. If there are no results in the row (pure fields after len(cols)), 
    it takes the attributes from the columns and calls the script command until all result fields are filled with step n_cols.
    """
    if not random_state:
        random_state = random.randint(1, 1000)
    else:
        random_state = random_state
    table = pd.read_csv(experiments_file)
    keys = ["--data_test ", "--load_model ", "-o ", "-c ", "--fingerprint ", "--n_bits ", "--n_cv ", "--n_jobs ", "-p ", "-g ", "--dummy ", "--n_iter ", "--features ", "--metric ", "--split "]
    params = ["Data test", "Load model", "Output", "Configs", "Fingerprint", "n_bits", "n_cv", "n_jobs", "Patience", "Gridsearch", "Dummy", "n_iter", "Features", "Metric", "Split"]
    pos_params = ['Model', 'Data train', 'Section']
    for i in range(table.shape[0]):
        rparams = False
        command = ""
        for p in pos_params:
            command += str(table[p][i]) + " "
        
        for c, p in enumerate(params):
            if not isnan(table[p][i]):
                if keys[c] in ["-g ", "--dummy "]:
                    command += keys[c] + " "
                else:
                    command += keys[c] + str(table[p][i]) + " "
        
        command += "-e " + experiments_file

        print(command)
        
        for j in range(int((table.shape[1] - len(params) - len(pos_params) / n_cols))):
            accuracy_test = accuracy_train = rec = auc = f1 = '-'
            command = command + " -t " + str(j)
            command_f = command.split()
            print(command_f)
            if isnan(table.iloc[i, j*n_cols + len(params) + len(pos_params)]):    
                if common_gridsearch:
                    accuracy_test, accuracy_train, rec, auc, f1, rparams = script(command_f, random_state, rparams)
                else:
                    accuracy_test, accuracy_train, rec, auc, f1, rparams = script(command_f, random_state, False)

                table = pd.read_csv(experiments_file)
                table.iloc[i, j*n_cols+len(params) + len(pos_params)] = rec[0]
                table.iloc[i, j*n_cols+1+len(params) + len(pos_params)] = rec[1]
                table.iloc[i, j*n_cols+2+len(params) + len(pos_params)] = auc
                table.to_csv(experiments_file, index=False)  # add results to experiments table


if __name__ == "__main__":
    n_cols = 3  # n columns of results per target in table
    common_gridsearch = True # one gridsearch for all experiments in row
    random_state = 66 # random state of all random in script
    def_experiments_file = 'etc/experiments_tox21.csv' # path to experiments table
    #experiments_file = input('Enter the image address (default is ' + def_experiments_file + '): ')
    #if experiments_file == '': experiments_file = def_experiments_file
    experiments_file = def_experiments_file
    
    main(experiments_file, common_gridsearch, random_state, n_cols)
