import numpy as np
import pandas as pd
import logging
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.model_selection import train_test_split

def get_data_bio_csv(filename, input_shape, DUMMY, MACCS, Morgan):
    data = pd.read_csv(filename)
    if "_morgan.csv" in filename:
        print("Loading data")
        logging.info("Loading data")
        if DUMMY: 
            data = np.array(data[:100])
        else:
            data = np.array(data)
        
        features = data[:,0:1024]
        labels = data[:,1024:]      
        
    elif "_maccs.csv" in filename:
        logging.info("Loading data")
        if DUMMY: 
            data = np.array(data[:100])
        else:
            data = np.array(data)
            
        features = data[:,0:167]
        labels = data[:,167:]
                
    else:
        smiles = data["smiles"]
        
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
            featurized = np.c_[features, labels]    
            np.savetxt(filename.replace(".csv", "_maccs.csv"), featurized, delimiter=",", fmt='%3f')
        if Morgan:
            features = [AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024) for x in ms if x]
            features = np.array(features)
            featurized = np.c_[features, labels]
            np.savetxt(filename.replace(".csv", "_morgan.csv"), featurized, delimiter=",", fmt='%3f')
    
        if DUMMY: 
            features = np.array(features[:100])
            labels = np.array(labels[:100])
    
    print("Data shape:",data.shape)
    logging.info("Data shape:",data.shape)
    print("Features shape:",features.shape)
    logging.info("Features shape:",features.shape)
    print("Labels shape:",labels.shape)
    logging.info("Labels shape:",labels.shape)
    print("Data loaded")
    logging.info("Data loaded")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=43)
    output_shape = labels.shape[1]
            
    return x_train, x_test, y_train, y_test, output_shape
