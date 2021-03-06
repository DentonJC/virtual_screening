#!/usr/bin/env python

"""
https://github.com/kudkudak

Featurization using rdkit descriptors.

Trivial wrapper around rdkit.Chem.Descriptors that includes embedding procedure

See main for example usage.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import re


def _camel_to_snail(s):
    """ Convert CamelCase to snail_case. """
    return re.sub('(?!^)([A-Z]+)', r'_\1', s).lower()

DESCRIPTORS = {_camel_to_snail(s): f for (s, f) in Descriptors.descList}

def rdkit_fetures_names():
    DESCRIPTORS = {_camel_to_snail(s): f for (s, f) in Descriptors.descList}
    return DESCRIPTORS.keys()


def _rdkit_transform(mol):
    res = []
    for (n, f) in DESCRIPTORS.items():
        try:
            res.append(f(mol))
        except ValueError:
            # return res.append(np.NaN)
            res.append(np.NaN)

    return np.array(res)


def smiles_to_rdkit(x):
    """
    Featurizes pd Series of smiles using rdkit.Descriptors

    Note - does not parallelize well, please use parallelization outside.
    Note - it has to minimize conformer, uses rdkit for that

    Params
    ------
    x: pd.Series
        Each value is smiles

    Returns
    -------
    output: pd.DataFrame
        Aligned with series rows output of computation. Fills with NaN failed computation (either due to cxcalc or conformation
        calculation)

    Notes
    -----
    Does not parallelize well, please use parallelization outside.
    """
    assert isinstance(x, pd.Series)

    missing = []
    features_index = []
    features_values = []

    for key, smi in x.items():
        try:
            # Try optimizing
            m = Chem.MolFromSmiles(smi)
            # m.UpdatePropertyCache(strict=False)

            confId = AllChem.EmbedMolecule(m, ignoreSmoothingFailures=True)
            if confId == -1:
                _ = AllChem.EmbedMolecule(m, useRandomCoords=True, ignoreSmoothingFailures=True)

            # Try logarithmically more iteration (still fails sometimes)
            for maxIters in [200, 2000, 20000, 200000, 2000000]:
                ret = AllChem.UFFOptimizeMolecule(m, maxIters=maxIters)
                if ret == 0:
                    break

            if ret != 0:
                missing.append(key)
                features_index.append(key)
                features_values.append(np.asarray([0]*len(DESCRIPTORS.keys())))
            else:
                features_index.append(key)
                features_values.append(_rdkit_transform(m))
        except:
            missing.append(key)
            features_index.append(key)
            features_values.append(np.asarray([0]*len(DESCRIPTORS.keys())))

    # print("Serialized {} out of {} compounds to sdf".format(len(x) - len(missing), len(x)))

    # Featurize and fill with NaNs
    #print(features_values)
    #try:
    #    print(len(features_values))
    #except:
    #    print(features_values.shape)
    #features_values = np.asarray(features_values)
    features = pd.DataFrame(features_values)#, index=features_index)
    #print(features.shape)
    #print(len(DESCRIPTORS.keys()))
    features.columns = DESCRIPTORS.keys()
    #features.reindex(features.index.union(missing))

    # if len(set(features.index)) != len(x):
    #    print("Missed compounds")

    return [features, missing]


if __name__ == "__main__":
    features = smiles_to_rdkit(pd.Series({"lol": "CC", "lol2": "CCN"}))
    print(features)
