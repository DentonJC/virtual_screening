#!/usr/bin/env python

"""
An additional script to run a series of experiments described in the .csv file.
"""
import os
import math
import argh
import pandas as pd

n_cols = 2  # 2 columns (train_acc, test, acc) per target


def isnan(x):
    """ Checks if the variable is empty (NaN and float). """
    return isinstance(x, float) and math.isnan(x)


def main(experiments_file='etc/experiments_muv.csv'):
    table = pd.read_csv(experiments_file)
    keys = ["--features ", "-o ", "-c ", "--fingerprint ", "--n-bits ", 
           "--n-jobs ", "-p ", "-g ", "--dummy "]
    cols = ["Address", "Data", "Section", "Features", "Output", "Configs",
            "Fingerprint", "n_bits", "n_jobs", "Patience", "Gridsearch", "Dummy"]
    for i in range(table.shape[0]):
        command = ""
        c = -(len(cols) - len(keys) + 1)
        for j in range(len(cols)):
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
        print(command)

        for j in range(int((table.shape[1] - len(cols)) / n_cols)):
            command_f = command + " -t " + str(j)
            text_file = open("tmp.sh", "w")
            text_file.write(command_f)
            text_file.close()
            try:
                os.system("nice -n 9 parallel -a tmp.sh -j 4")
            except:
                print("Something wrong with experiment")
            os.remove("tmp.sh")
            
            res_file = open(os.path.dirname(os.path.realpath(__file__)) + "/tmp/last_result","r")
            res = []
            for r in res_file:
                res.append(r)
            table.iloc[i, j*n_cols+len(cols)] = res[0].replace('\n','')
            table.iloc[i, j*n_cols+1+len(cols)] = res[1].replace('\n','')
            table.to_csv(experiments_file, index=False)

parser = argh.ArghParser()
argh.set_default_command(parser, main)

if __name__ == "__main__":
    parser.dispatch()
