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
            for maxIters in [200, 2000, 20000, 200000]:
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

    if len(set(features.index)) == len(x):
        print("Missed compounds")

    return features, missing


if __name__ == "__main__":
    features = smiles_to_desc_rdkit(pd.Series({"lol": "CC", "lol2": "CCN"}))
    print(features)
