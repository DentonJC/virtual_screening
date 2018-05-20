"""
Various splitters
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles, include_chirality=False):
  """
  Compute the Bemis-Murcko scaffold for a SMILES string.

  Notes
  -----
  Copied from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
  """
  mol = Chem.MolFromSmiles(smiles)
  scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality)
  return scaffold

def scaffold_split(smiles, frac_train=.8, seed=777, log_every_n=1000):
    """
    Splits compounds into train/validation/test by scaffold.

    Params
    ------
    smiles: list
        List of smiles
    frac_train: float, default: 0.8
        Float in [0, 1] range indicating size of the training set
    seed: int
        Used to shuffle smiles before splitting

    Notes
    -----
    Copied from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    """
    smiles = list(smiles)

    scaffolds = {}
    logger.debug("About to generate scaffolds")
    data_len = len(smiles)

    rng = np.random.RandomState(seed)
    rng.shuffle(smiles)

    for ind, smi in enumerate(smiles):
        if ind % log_every_n == 0:
            logger.debug("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = generate_scaffold(smi)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)
    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    train_cutoff = frac_train * len(smiles)
    train_inds, test_inds = [], []
    logger.debug("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) > train_cutoff:
            test_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, test_inds

if __name__ == "__main__":
    # Test scaffold splitting
    # TODO: Problem for single chain molecules?
    # NOTE: Last molecule is viagra
    smiles = ['CC(C)(N)Cc1ccccc1', 'CC(C)(Cl)Cc1ccccc1', 'c1ccccc1',
        'CCc1nn(C)c2c(=O)[nH]c(nc12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4']
    splits = scaffold_split(smiles, frac_train=0.5)
    assert splits[0] == [0, 1, 2] or splits[1] == [0, 1, 2], "Correctly separated scaffolds"
    print(splits)