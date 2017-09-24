import numpy as np
import pandas as pd
import logging
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.model_selection import train_test_split
from src._desc_rdkit import smiles_to_desc_rdkit
from src.main import drop_nan

Dummy_n = 10000
number_of_features_physical_representaion = 196

def get_data(filename, DUMMY, MACCS, Morgan, nBits, set_targets, set_features):
    data = pd.read_csv(filename)
    smiles = []
    if DUMMY: 
        data = data[:Dummy_n]
    
    if "_morgan" in filename:
        print("Loading data")
        logging.info("Loading data")
        if DUMMY: 
            data = np.array(data[:Dummy_n])
        else:
            data = np.array(data)
        
        features = data[:,0:nBits+number_of_features_physical_representaion]
        labels = data[:,nBits+number_of_features_physical_representaion:]      
        
    elif "_maccs" in filename:
        print("Loading data")
        logging.info("Loading data")
        if DUMMY: 
            data = np.array(data[:Dummy_n])
        else:
            data = np.array(data)
            
        features = data[:,0:167+number_of_features_physical_representaion]
        labels = data[:,167+number_of_features_physical_representaion:]
    
    else:
        print("Physic data extraction")
        smiles = data["smiles"]
        physic_smiles = pd.Series(smiles)
        physic_data = smiles_to_desc_rdkit(physic_smiles)
        data = np.array(data)
        _, cols = data.shape
        l = []
        for c in range(cols):
            if data[0][c] in (0, 1) or pd.isnull(data[0][c]):
                l.append(data[:,c])

        labels = np.array(l).T
        
        print("Featurization")
        logging.info("Featurization")
        ms = [Chem.MolFromSmiles(x) for x in smiles]
        if MACCS:
            features = [MACCSkeys.GenMACCSKeys(x) for x in ms if x]
            features = np.array(features)
            features = np.c_[features, physic_data]
            featurized = np.c_[features, labels]    
            np.savetxt(filename.replace(".csv", "_maccs.csv"), featurized, delimiter=",", fmt='%3f')
        if Morgan:
            features = [AllChem.GetMorganFingerprintAsBitVect(x,3,nBits=nBits) for x in ms if x]
            features = np.array(features)
            features = np.c_[features, physic_data]
            featurized = np.c_[features, labels]
            np.savetxt(filename.replace(".csv", "_morgan_"+str(nBits)+".csv"), featurized, delimiter=",", fmt='%3f')
    
        if DUMMY: 
            features = np.array(features[:Dummy_n])
            labels = np.array(labels[:Dummy_n])
    # remove prints
    print("Data shape:", str(data.shape))
    logging.info("Data shape: %s", str(data.shape))
    print("Features shape:", str(features.shape))
    logging.info("Features shape: %s", str(features.shape))
    print("Labels shape:", str(labels.shape))
    logging.info("Labels shape: %s", str(labels.shape))
    print("Data loaded")
    logging.info("Data loaded")

    x_train, x, y_train, y = train_test_split(features, labels, test_size=0.4, random_state=43)
    x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=0.5, random_state=43)
    
    if set_targets:
        y_train = y_train[:,set_targets].reshape(y_train.shape[0],len(set_targets))
        y_test = y_test[:,set_targets].reshape(y_test.shape[0],len(set_targets))
        y_val = y_val[:,set_targets].reshape(y_val.shape[0],len(set_targets))
        
    if set_features:
        x_train = x_train[:,set_features].reshape(x_train.shape[0],len(set_features))
        x_test = x_test[:,set_features].reshape(x_test.shape[0],len(set_features))
        x_val = x_val[:,set_features].reshape(x_val.shape[0],len(set_features))
    
    x_train, y_train = drop_nan(x_train, y_train)
    x_test, y_test = drop_nan(x_test, y_test)
    x_val, y_val = drop_nan(x_val, y_val)
    # remove prints
    print("X_train:", x_train.shape)
    print("Y_train:", y_train.shape)
    print("X_test:", x_test.shape)
    print("Y_test:", y_test.shape)
    print("X_val:", x_val.shape)
    print("Y_val:", y_val.shape)
    logging.info("X_train: %s", str(x_train.shape))
    logging.info("Y_train: %s", str(y_train.shape))
    logging.info("X_test: %s", str(x_test.shape))
    logging.info("Y_test: %s", str(y_test.shape))
    logging.info("X_val: %s", str(x_val.shape))
    logging.info("Y_val: %s", str(y_val.shape))
    _, input_shape = x_train.shape
    _, output_shape = y_train.shape
            
    return x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles
