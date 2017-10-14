import csv
import pandas as pd
import sys
import math


def isnan(x):
    return isinstance(x, float) and math.isnan(x)


def write_experiment(train_acc, test_acc, target_name, experiments_file):
    try:
        arguments = ""
        for arg in sys.argv:
            arguments+=arg+" "

        table = pd.read_csv(experiments_file)
        for i in range(table.shape[0]):
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
                command+="--dummy "
            if command in arguments:
                i+=1 # becouse pd.read i != csv.reader
                r = csv.reader(open(experiments_file)) # Here your csv file
                lines = [l for l in r]
                lines[i][12 + target_name[0]*2] = str(train_acc)
                lines[i][12 + 1 + target_name[0]*2] = str(test_acc)
                writer = csv.writer(open(experiments_file, 'w'))
                writer.writerows(lines)
                print("Write results of experiment")
            
    except:
        print("Can't write results of experiment")
