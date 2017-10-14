#!/usr/bin/env python

import os
import math
import argh
import pandas as pd
import numpy as np


def isnan(x):
    return isinstance(x, float) and math.isnan(x)

def main(experiments_file = 'experiments.csv'):
    table = pd.read_csv(experiments_file)
    for i in range(table.shape[0]):
        command = ""
        if not isnan(table["Address"][i]): 
            command+=str(table["Address"][i]) + " "
        if not isnan(table["Data"][i]): 
            command+=table["Data"][i] + " "
        if not isnan(table["Section"][i]): 
            command+=str(table["Section"][i]) + " "
        if not isnan(table["Features"][i]): 
            command+="--features " + str(table["Features"][i]) + " "
        if not isnan(table["Output"][i]): 
            command+="-o " + str(table["Output"][i]) + " "
        if not isnan(table["Configs"][i]): 
            command+="-c " + str(table["Configs"][i]) + " "
        if not isnan(table["Fingerprint"][i]): 
            command+="--fingerprint " + str(table["Fingerprint"][i]) + " "
        if not isnan(table["n_bits"][i]): 
            command+="--n-bits " + str(int(table["n_bits"][i])) + " "
        if not isnan(table["n_jobs"][i]): 
            command+="--n-jobs " + str(int(table["n_jobs"][i])) + " "
        if not isnan(table["Patience"][i]): 
            command+="-p " + str(int(table["Patience"][i])) + " "
        if not isnan(table["Gridsearch"][i]): 
            command+="-g "
        if not isnan(table["Dummy"][i]): 
            command+="--dummy "

        command+="-e " + experiments_file
        for j in range(int((table.shape[1] - 12) / 2)): # 12 - num of parameters, 4 columns per target
            command_f = command + " -t " + str(j)
            text_file = open("tmp.sh", "w")
            text_file.write(command_f)
            text_file.close()
            try:
                os.system("nice -n 9 parallel -a tmp.sh -j -1")
            except:
                print("Something wrong with experiment")
            os.remove("tmp.sh")

parser = argh.ArghParser()
argh.set_default_command(parser, main)

if __name__ == "__main__":
    parser.dispatch()

