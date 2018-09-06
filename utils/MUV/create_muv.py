#!/usr/bin/python3

"""
Building dataset from the results of experiments https://jcheminf.springeropen.com/articles/10.1186/1758-2946-5-26.

First, a list of unique SMILES is compiled, and then each two files from the f_list form one dataset column,
if the value of SMILES is in the file with the end of "decoys", then its activity is 0,
if in the "actives" - 1, else - nan.
"""

import pandas as pd
import numpy as np

output = "muv.csv"

# unsorted
# f_list = os.listdir(os.path.dirname(os.path.realpath(__file__)))
# f_list.remove("create_muv.py")
# f_list.remove(output)

# sorted
f_list = [
    'cmp_list_MUV_466_actives.dat',
    'cmp_list_MUV_466_decoys.dat',
    'cmp_list_MUV_548_actives.dat',
    'cmp_list_MUV_548_decoys.dat',
    'cmp_list_MUV_600_actives.dat',
    'cmp_list_MUV_600_decoys.dat',
    'cmp_list_MUV_644_actives.dat',
    'cmp_list_MUV_644_decoys.dat',
    'cmp_list_MUV_652_actives.dat',
    'cmp_list_MUV_652_decoys.dat',
    'cmp_list_MUV_689_actives.dat',
    'cmp_list_MUV_689_decoys.dat',
    'cmp_list_MUV_692_actives.dat',
    'cmp_list_MUV_692_decoys.dat',
    'cmp_list_MUV_712_actives.dat',
    'cmp_list_MUV_712_decoys.dat',
    'cmp_list_MUV_713_actives.dat',
    'cmp_list_MUV_713_decoys.dat',
    'cmp_list_MUV_733_actives.dat',
    'cmp_list_MUV_733_decoys.dat',
    'cmp_list_MUV_737_actives.dat',
    'cmp_list_MUV_737_decoys.dat',
    'cmp_list_MUV_810_actives.dat',
    'cmp_list_MUV_810_decoys.dat',
    'cmp_list_MUV_832_actives.dat',
    'cmp_list_MUV_832_decoys.dat',
    'cmp_list_MUV_846_actives.dat',
    'cmp_list_MUV_846_decoys.dat',
    'cmp_list_MUV_852_actives.dat',
    'cmp_list_MUV_852_decoys.dat',
    'cmp_list_MUV_858_actives.dat',
    'cmp_list_MUV_858_decoys.dat',
    'cmp_list_MUV_859_actives.dat',
    'cmp_list_MUV_859_decoys.dat'
]

print("Creating list of smiles")

smiles = []

for f in f_list:
    print(f)
    data = pd.read_csv(f, sep='\t')
    data = np.array(data)
    for d in data[:, 2]:
        if d not in smiles:
            smiles.append(d)

np_smiles = np.array(smiles)

print("Creating dataset")
table = np.zeros(shape=(len(smiles), int(len(f_list) / 2)))
table.fill(np.nan)
table = np.c_[np_smiles, table]

column = 1
for f in f_list:
    print(f)
    data = pd.read_csv(f, sep='\t')
    data = np.array(data)
    for d in data[:, 2]:
        string = smiles.index(d)
        if "actives" in f:
            table[string][int(column)] = 1
        if "decoys" in f:
            table[string][int(column)] = 0
    column += 0.5

print(table.shape)
np.savetxt(output, table, delimiter=",", fmt='%s')
