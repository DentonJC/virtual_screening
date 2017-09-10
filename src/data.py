import numpy as np
import pandas as pd
import logging
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.model_selection import train_test_split
from src._desc_rdkit import smiles_to_desc_rdkit


def get_data_bio_csv(filename, input_shape, DUMMY, MACCS, Morgan, nBits=1024):
    data = pd.read_csv(filename)
    if DUMMY: 
        data = data[:100]
    print(data.shape)
    if "_morgan.csv" in filename:
        print("Loading data")
        logging.info("Loading data")
        if DUMMY: 
            data = np.array(data[:100])
        else:
            data = np.array(data)
        
        features = data[:,0:nBits+196]
        labels = data[:,nBits+196:]      
        
    elif "_maccs.csv" in filename:
        logging.info("Loading data")
        if DUMMY: 
            data = np.array(data[:100])
        else:
            data = np.array(data)
            
        features = data[:,0:167+196]
        labels = data[:,167+196:]
    
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
            features = np.array(features[:100])
            labels = np.array(labels[:100])
    
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
    output_shape = labels.shape[1]
            
    return x_train, x_test, x_val, y_train, y_test, y_val, output_shape, smiles
