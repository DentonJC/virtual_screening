"""
Various splitters
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from tqdm import tqdm

def generate_scaffold(smiles, include_chirality=False):
  """
  Compute the Bemis-Murcko scaffold for a SMILES string.

  Notes
  -----
  Copied from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
  """
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
      logger.warning("Failedcalculating scaffold for " + smiles)
      return 'fail'
  scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality)
  return scaffold


def scaffold_split(smiles, frac_train=.8, seed=777):
    """
    Splits compounds into train/validation/test by scaffold.

    Warning: if there is one very popular scaffold can produce unbalanced split

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

    for ind, smi in tqdm(enumerate(smiles), total=len(smiles)):
        scaffold = generate_scaffold(smi)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    scaffolds_keys = list(scaffolds)
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffolds_keys)

    train_cutoff = frac_train * len(smiles)
    train_inds, test_inds = [], []
    for scaffold_key in scaffolds_keys:
        if len(train_inds) > train_cutoff:
            test_inds += scaffolds[scaffold_key]
        else:
            train_inds += scaffolds[scaffold_key]
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