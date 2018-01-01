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
    keys = ["-o ", "-c ", "--fingerprint ", "--n_bits ", "--n_jobs ", "-p ", "-g ", "--dummy ", "--n_iter ", "--features "]
    cols = ["Model", "Data test", "Data train", "Section", "Features", "Output", "Configs",
            "Fingerprint", "n_bits", "n_jobs", "Patience", "Gridsearch", "Dummy", "n_iter"]
    for i in range(table.shape[0]):
        rparams = False
        command = str(table[cols[0]][i]) + " "
        c = -(len(cols) - len(keys) + 1)
        for j in range(1, len(cols)):
            c+=1
            if not isnan(table[cols[j]][i]):
                if j >= len(cols) - len(keys):
                    if keys[c] in ["-g ", "--dummy "]:
                        command += keys[c] + " "
                    else:
                        command += keys[c] + str(table[cols[j]][i]) + " "
                else:
                    command += str(table[cols[j]][i]) + " "
        command += "-e " + experiments_file

        for j in range(int((table.shape[1] - len(cols)) / n_cols)):
            accuracy_test = accuracy_train = rec = auc = f1 = '-'
            command = command + " -t " + str(j)
            command_f = command.split()
            print(command_f)
            if isnan(table.iloc[i, j*n_cols+len(cols)]):    
                if common_gridsearch:
                    accuracy_test, accuracy_train, rec, auc, f1, rparams = script(command_f, random_state, rparams)
                else:
                    accuracy_test, accuracy_train, rec, auc, f1, rparams = script(command_f, random_state, False)
                try:
                    rparams = rparams['params'][0]
                except (KeyError, AttributeError):
                    pass

                table = pd.read_csv(experiments_file)
                table.iloc[i, j*n_cols+len(cols)] = rec[0]
                table.iloc[i, j*n_cols+1+len(cols)] = rec[1]
                table.iloc[i, j*n_cols+2+len(cols)] = rec[0] + rec[1]
                table.iloc[i, j*n_cols+3+len(cols)] = auc
                table.to_csv(experiments_file, index=False)  # add results to experiments table


if __name__ == "__main__":
    n_cols = 4  # n columns of results per target in table
    common_gridsearch = False # one gridsearch for all experiments in row
    random_state = 66 # random state of train-test split

    def_experiments_file = 'etc/experiments_tox21.csv' # path to experiments table
    experiments_file = input('Enter the image address (default is ' + def_experiments_file + '): ')
    if experiments_file == '': experiments_file = def_experiments_file
    
    main(experiments_file, common_gridsearch, random_state, n_cols)
