import pandas as pd
import numpy as np
import math
import os

TABLE_NAME = "experiments.csv"

def isnan(x):
    return isinstance(x, float) and math.isnan(x)

table = pd.read_csv(TABLE_NAME)
for i in range(table.shape[0]):
    if (isnan(table["Train acc"][i])) or (isnan(table["Test acc"][i])):
        command = ""
        if not isnan(table["Address"][i]): 
            command+=str(table["Address"][i]) + " "
        if not isnan(table["Data"][i]): 
            command+=str(table["Data"][i]) + " "
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
            command+="--n-bits " + str(table["n_bits"][i]) + " "
        if not isnan(table["n_jobs"][i]): 
            command+="--n-jobs " + str(table["n_jobs"][i]) + " "
        if not isnan(table["Patience"][i]): 
            command+="-p " + str(table["Patience"][i]) + " "
        if not isnan(table["Gridsearch"][i]): 
            command+="-g "
        if not isnan(table["Dummy"][i]): 
            command+="--dummy"

        text_file = open("tmp.sh", "w")
        text_file.write(command)
        text_file.close()
        try:
            os.system("nice -n 9 parallel -a tmp.sh -j 10")
        except:
            print("Something wrong with experiment")
        os.remove("tmp.sh")

