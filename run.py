#!/usr/bin/env python

"""
An additional script to run a series of experiments described in the .csv file.
"""
import os
import math
import random
import pandas as pd
from src.run_model import script


def isnan(x):
    """ Checks if the variable is empty (NaN and float). """
    return isinstance(x, float) and math.isnan(x)


def main(experiments_file, common_gridsearch, skip_error, random_state, n_cols):
    if not random_state:
        random_state = random.randint(1, 1000)
    else:
        random_state = random_state
    table = pd.read_csv(experiments_file)
    keys = ["-o ", "-c ", "--fingerprint ", "--n_bits ", "--n_jobs ", "-p ", "-g ", "--dummy ", "--n_iter ", "--features "]
    cols = ["Model", "Data", "Section", "Features", "Output", "Configs",
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
                        command += keys[c] + " "#" True "                        
                    else:
                        command += keys[c] + str(table[cols[j]][i]) + " "
                else:
                    command += str(table[cols[j]][i]) + " "
        command += "-e " + experiments_file

        for j in range(int((table.shape[1] - len(cols)) / n_cols)):
            command = command + " -t " + str(j)
            command_f = command.split()
            print(command_f)
            if isnan(table.iloc[i, j*n_cols+len(cols)]):
                if skip_error:
                    if common_gridsearch:
                        r_test, r_train, rparams, rec = script(command_f, random_state, rparams)
                    else:
                        r_test, r_train, rparams, rec = script(command_f, random_state, False)
                    try:
                        rparams = rparams['params'][0]
                    except KeyError:
                        pass
                else:
                    try:
                        if common_gridsearch:
                            r_test, r_train, rparams, rec = script(command_f, random_state, rparams)
                        else:
                            r_test, r_train, rparams, rec = script(command_f, random_state, False)
                        try:
                            rparams = rparams['params'][0]
                        except (KeyError, AttributeError):
                            pass
                    except:
                        res = (0,0)
                table.iloc[i, j*n_cols+len(cols)] = rec[0]
                table.iloc[i, j*n_cols+1+len(cols)] = rec[1]
                table.to_csv(experiments_file, index=False)


if __name__ == "__main__":
    # becouse of argparse script must be started without arguments
    n_cols = 2  # n columns of results per target in table
    experiments_file = 'etc/experiments_tox21.csv' # path to experiments table
    common_gridsearch = False # one gridsearch for all experiments in row
    skip_error = True # ontinue the experiments after the error
    random_state = 478 # random state of train-test split
    main(experiments_file, common_gridsearch, skip_error, random_state, n_cols)
