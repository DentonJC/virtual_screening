#!/usr/bin/env python

"""
An additional script to run a series of experiments described in the .csv file.
"""
import os
import logging
import math
import argparse
import pandas as pd
from src.run_model import script

n_cols = 2  # 2 columns (train_acc, test, acc) per target


def isnan(x):
    """ Checks if the variable is empty (NaN and float). """
    return isinstance(x, float) and math.isnan(x)

def get_options():
    parser = argparse.ArgumentParser(prog="logreg.py data section")
    parser.add_argument('experiments_file', nargs='?', default='etc/experiments_tox21_NR-AR.csv', help='path to experiments table')
    return parser

def main():
    options = get_options().parse_args()
    table = pd.read_csv(options.experiments_file)
    keys = ["-o ", "--dummy ", "--fingerprint ", "--n_bits ", "--n_jobs ", "-p ", "-g ", "-c ", "--n_iter ", "--features "]
    cols = ["Model", "Data", "Section", "Features", "Output", "Configs",
            "Fingerprint", "n_bits", "n_jobs", "Patience", "Gridsearch", "Dummy", "n_iter"]
    for i in range(table.shape[0]):
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
        command += "-e " + options.experiments_file

        for j in range(int((table.shape[1] - len(cols)) / n_cols)):
            command = command + " -t " + str(j)
            command_f = command.split()
            print(command_f)
            if isnan(table.iloc[i, j*n_cols+len(cols)]):
                try:
                    res = script(command_f)
                except:
                    res = (0,0)
                table.iloc[i, j*n_cols+len(cols)] = res[0]
                table.iloc[i, j*n_cols+1+len(cols)] = res[1]
                table.to_csv(options.experiments_file, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
