# import logging
import multiprocessing
import numpy as np
import pandas as pd
# from rdkit.Chem import AllChem
from tqdm import tqdm 
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from joblib import Parallel, delayed
from joblib.parallel import BACKENDS
from moloi.descriptors.rdkit_descriptor import smiles_to_rdkit
from moloi.descriptors.mordred_descriptor import smiles_to_mordred
from moloi.descriptors.morgan_descriptor import smiles_to_morgan
from moloi.descriptors.spectrophore_descriptor import smiles_to_spectrophore

BACKEND = 'loki'
if BACKEND not in BACKENDS.keys():
    BACKEND = 'multiprocessing'


def descriptor_rdkit(logger, smiles, verbose, n_jobs):
    logger.info("RDKit data extraction")
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if len(smiles) < n_jobs:
        n_jobs = len(smiles)

    parallel = []
    parallel.append(Parallel(n_jobs=n_jobs, backend=BACKEND, verbose=0)(delayed(smiles_to_rdkit)(pd.Series(s)) for s in tqdm(smiles)))
    parallel = np.array(parallel)
    physic_data = parallel[0][0][0].T
    missing = np.array(parallel[0][0][1])
    for i in range(1, len(smiles)):
        physic_data = np.c_[physic_data, parallel[0][i][0].T]
        missing = np.append(missing, parallel[0][i][1])
    rdkit_data = pd.DataFrame(physic_data.T)

    return missing, rdkit_data


def descriptor_mordred(logger, smiles, verbose, n_jobs):
    logger.info("Mordred data extraction")
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if len(smiles) < n_jobs:
        n_jobs = len(smiles)

    parallel = []
    parallel.append(Parallel(n_jobs=n_jobs, backend=BACKEND, verbose=0)(delayed(smiles_to_mordred)(pd.Series(s)) for s in tqdm(smiles)))
    parallel = np.array(parallel)
    physic_data = parallel[0][0][0].T
    missing = np.array(parallel[0][0][1])
    for i in range(1, len(smiles)):
        physic_data = np.c_[physic_data, parallel[0][i][0].T]
        missing = np.append(missing, parallel[0][i][1])
    physic_data = pd.DataFrame(physic_data.T)

    return missing, physic_data


def descriptor_maccs(logger, smiles):
    logger.info("Maccs data extraction")
    ms = [Chem.MolFromSmiles(x) for x in smiles]
    features = [MACCSkeys.GenMACCSKeys(x) for x in ms if x]
    return np.array(features)


def descriptor_morgan(logger, smiles, n_bits, hashed=True, radius=2):
    logger.info("Morgan data extraction")
    features = [smiles_to_morgan(x, hashed=hashed, radius=radius, n_bits=n_bits) for x in smiles if x]
    return np.array(features)


def descriptor_spectrophore(logger, smiles, n_bits, n_jobs, verbose):
    logger.info("Spectrophore data extraction")
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if len(smiles) < n_jobs:
        n_jobs = len(smiles)
    features = []
    features.append(Parallel(n_jobs=n_jobs, backend=BACKEND, verbose=0)(delayed(smiles_to_spectrophore)(i, n_samples=n_bits) for i in tqdm(smiles)))
    features = np.array(features)
    features = [i for sub in features for i in sub]
    return np.array(features)
