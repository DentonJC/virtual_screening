#!/usr/bin/env python

"""
Featurization using rdkit descriptors.

Trivial wrapper around rdkit.Chem.Descriptors that includes embedding procedure

See main for example usage.
"""
import tqdm
import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from mordred import Calculator, descriptors


def _camel_to_snail(s):
    """ Convert CamelCase to snail_case. """
    return re.sub('(?!^)([A-Z]+)', r'_\1', s).lower()

DESCRIPTORS = {_camel_to_snail(s): f for (s, f) in Descriptors.descList}


def _rdkit_transform(mol):
    res = []
    for (n, f) in DESCRIPTORS.items():
        try:
            res.append(f(mol))
        except ValueError:
            return res.append(np.NaN)

    return np.array(res)


def smiles_to_desc_rdkit(x):
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

    for key, smi in tqdm.tqdm(x.items(), total=len(x)):
        try:
            # Try optimizing
            m = Chem.MolFromSmiles(smi)
            # m.UpdatePropertyCache(strict=False)

            confId = AllChem.EmbedMolecule(m, ignoreSmoothingFailures=True)
            if confId == -1:
                _ = AllChem.EmbedMolecule(m, useRandomCoords=True, ignoreSmoothingFailures=True)

            # Try logarithmically more iteration (still fails sometimes)
            for maxIters in [200, 2000, 20000, 200000, 2000000, 20000000, 2000000000]:
                ret = AllChem.UFFOptimizeMolecule(m, maxIters=maxIters)
                if ret == 0:
                    break

            if ret != 0:
                missing.append(key)
            else:
                features_index.append(key)
                features_values.append(_rdkit_transform(m))
        except:
            missing.append(key)

    print("Serialized {} out of {} compounds to sdf".format(len(x) - len(missing), len(x)))

    # Featurize and fill with NaNs
    features = pd.DataFrame(features_values, index=features_index)
    features.columns = DESCRIPTORS.keys()
    features.reindex(features.index.union(missing))

    if len(set(features.index)) != len(x):
        print("Missed compounds")
    
    return features, missing
    
    
def smiles_to_desc_mordred(x):
    """
    Featurizes pd Series of smiles using mordred.Calculator

    Params
    ------
    x: pd.Series
        Each value is smiles

    Returns
    -------
    output: pd.DataFrame
    """
    assert isinstance(x, pd.Series)
    
    missing = []
    features_index = []
    features_values = []
    molecule_values = []

    for key, smi in tqdm.tqdm(x.items(), total=len(x)):
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                features_index.append(key)
                molecule_values.append(m)
            else:
                missing.append(key)
        except:
            missing.append(key)
        
    calc = Calculator(descriptors)
        
    features = calc.pandas(molecule_values)
    features = features.reindex(features.index.union(missing))
    #mask = features.applymap(lambda x: isinstance(x, (int, float))).values
    #features = features.where(mask)
    #features = features.dropna(axis=1,how='all')
    # features = features.convert_objects(convert_numeric=True) # string to NaN

    if len(set(features.index)) != len(x):
        print("Missed compounds")
    
    return features, missing#, calc.descriptors


if __name__ == "__main__":
    features = smiles_to_desc_rdkit(pd.Series({"lol": "CC", "lol2": "CCN"}))
    print(features)
