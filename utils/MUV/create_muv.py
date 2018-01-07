#!/usr/bin/python3

import pandas as pd
import numpy as np
import os

output = "muv.csv"

f_list = os.listdir(os.path.dirname(os.path.realpath(__file__)))
f_list.remove("create_muv.py")
f_list.remove(output)
print(f_list)
print("Creating list of smiles")
smiles = []
for f in f_list:
    print(f)
    data = pd.read_csv(f, sep='\t')
    data = np.array(data)
    for d in data[:,2]:
        if d not in smiles:
            smiles.append(d)
            #print(d)

np_smiles = np.array(smiles)
#np.savetxt("smiles.csv", np_smiles, delimiter=",")
#smiles = pd.read_csv("smiles.csv")
print("Creating dataset")
table = np.zeros(shape=(len(smiles),int(len(f_list)/2)))
table.fill(np.nan)
table = np.c_[np_smiles, table]

column = 1
for f in f_list:
    print(f)
    data = pd.read_csv(f, sep='\t')
    data = np.array(data)
    for d in data[:,2]:
        string = smiles.index(d)
        if "actives" in f:
            table[string][int(column)] = 1
        if "decoys" in f:
            table[string][int(column)] = 0
    column+=0.5

print(table.shape)
np.savetxt(output, table, delimiter=",", fmt='%s')


