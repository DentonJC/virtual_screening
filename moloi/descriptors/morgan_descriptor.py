#!/usr/bin/env python

"""
https://github.com/kudkudak
"""


import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_morgan(smi, hashed=True, radius=2, n_bits=300):
    mol = Chem.MolFromSmiles(smi)
    if hashed:
        try:
            vect = AllChem.GetHashedMorganFingerprint(mol=mol,
                                                      radius=radius,
                                                      nBits=n_bits)
            vect = vect.GetNonzeroElements()
            vect_keys = list(vect.keys())
            vect_values = list(vect.values())
            # Not sure how to transform it better
            vect_dense = np.zeros(shape=(n_bits,))
            vect_dense[vect_keys] = vect_values
            return vect_dense
        except:
            print("Failed computing morgan fingerprint for %s", smi)
            return np.zeros(shape=(n_bits,))
    else:
        try:
            mol = Chem.MolFromSmiles(smi)
            vect = AllChem.GetMorganFingerprintAsBitVect(mol=mol,
                                                         radius=radius,
                                                         nBits=n_bits)
            return np.array(vect)
        except:
            print("Failed computing morgan fingerprint for %s", smi)
            return np.zeros(shape=(n_bits,))


if __name__ == "__main__":
    features = smiles_to_morgan("CC")
    print(features)
