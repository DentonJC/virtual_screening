#!/usr/bin/env python

"""
Featurization using rdkit descriptors.

Trivial wrapper around rdkit.Chem.Descriptors that includes embedding procedure

See main for example usage.
"""
import tqdm
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors


# features names
def mordred_fetures_names():
    calc = Calculator(descriptors)
    cols = []
    for i in calc.descriptors:
        i = str(i)
        i = i.split(".")[-1]
        cols.append(str(i))
    return cols


def smiles_to_mordred(x, embed=True):
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
    all_features = []
    features_index = []
    molecule_values = []

    for key, smi in tqdm.tqdm(x.items(), total=len(x)):
        success = -1
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                m = Chem.AddHs(m)
                molecule_values.append(m)
            else:
                missing.append(key)

            if embed:
                # Taken from https://programtalk.com/python-examples/rdkit.Chem.AllChem.EmbedMolecule/
                success = AllChem.EmbedMolecule(m, useRandomCoords=False, ignoreSmoothingFailures=True)
                if success == -1:  # Failed
                    #print("Failed 1st embedding trial for " + smi)
                    success = AllChem.EmbedMolecule(m, useRandomCoords=True)
                if success == 0:
                    for maxIters in [200, 2000, 20000, 200000, 2000000]:
                        ret = AllChem.UFFOptimizeMolecule(m, maxIters=maxIters)
                        if ret == 0:
                            break
                else:
                    #print("Failed 2nd and last embedding trial for " + smi)
                    missing.append(key)
        except:
            missing.append(key)

        calc = Calculator(descriptors)
        
        try:
            try:
                features = calc(m, id=0)._values  # Call on the first conformer
            except:
                features = calc(m, id=0)
        except:
            features = [0]*1824

        if success == -1 and embed:
            all_features.append([0 for _ in features])
            missing.append(key)
        else:
            features_index.append(key)
            all_features.append(features)

    all_features = pd.DataFrame(all_features)
    cols = []
    for i in calc.descriptors:
        i = str(i)
        i = i.split(".")[-1]
        cols.append(str(i))

    all_features.columns = cols
    return [all_features, missing]


if __name__ == "__main__":
    features = smiles_to_mordred(pd.Series({"lol": "CC", "lol2": "CCN"}))
    print(features)
