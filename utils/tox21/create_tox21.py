#!/usr/bin/python3

import pandas as pd
import numpy as np
import os

output = "tox21.csv"

f_list = os.listdir(os.path.dirname(os.path.realpath(__file__)))
f_list.remove("create_tox21.py")
f_list.remove(output)
print(f_list)
print("Creating list of smiles")
smiles = []
for f in f_list:
    print(f)
    data = pd.read_csv(f, sep='\t')
    data = np.array(data)    
    for d in data[:,0]:
        if d not in smiles:
            smiles.append(d)

#np.savetxt("smiles.csv", np_smiles, delimiter=",")
#smiles = pd.read_csv("smiles.csv")
print("Creating dataset")
table = np.zeros(shape=(len(smiles),int(len(f_list))))
table.fill(np.nan)
column = 0
for f in f_list:
    print(f)
    data = pd.read_csv(f, sep='\t')
    data = np.array(data)
    for i, d in enumerate(data[:,0]):
        string = smiles.index(d)
        table[string][int(column)] = data[:,2][i]
    column+=1

table = np.c_[table, smiles]
print(table.shape)
np.savetxt(output, table, delimiter=",", fmt='%s')
