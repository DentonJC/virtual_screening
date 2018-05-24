import sys
from moloi.splits.scaffold_split import scaffold_split
from moloi.splits.cluster_split import cluster_split
from sklearn.model_selection import train_test_split


def create_cv(smiles, split_type, n_cv, random_state, y=False):
    if split_type == "scaffold":
        count = n_cv
        n_cv = [([], []) for _ in range(count)]
        for i in range(count):
            train, test = scaffold_split(smiles, frac_train = 1 - ((len(smiles) / count) / (len(smiles)/100))/100, seed=random_state)
            n_cv[i][0].append(train)
            n_cv[i][1].append(test)
    
    # The cluster size can not be specified.
    # elif split_type == "cluster":
    #     count = n_cv        
    #     n_cv = [([], []) for _ in range(count)]
    #     for i in range(count):
    #         train, test = cluster_split(smiles, test_cluster_id=1, n_splits=2)
    #         print(test)
    #         print(train)
    #         print(len(test))
    #         print(len(train))
    #         n_cv[i][0].append(train)
    #         n_cv[i][1].append(test)
    
            
    elif split_type == "random" or split_type == "cluster":
        count = n_cv        
        n_cv = [([], []) for _ in range(count)]
        for i in range(count):
            shuffle = True
            stratify = None
            idx = [i for i in range(len(smiles))]
            train, test = train_test_split(idx, test_size= 1 - ((len(smiles) / count) / (len(smiles)/100))/100, stratify=stratify, shuffle=shuffle)
            n_cv[i][0].append(train)
            n_cv[i][1].append(test)

    elif split_type == "stratified":
        count = n_cv        
        n_cv = [([], []) for _ in range(count)]
        for i in range(count):
            shuffle = True
            stratify = y
            idx = [i for i in range(len(smiles))]
            train, test = train_test_split(idx, test_size= 1 - ((len(smiles) / count) / (len(smiles)/100))/100, stratify=stratify, shuffle=shuffle)
            n_cv[i][0].append(train)
            n_cv[i][1].append(test)

    else:
        print("Wrong split type")
        sys.exit(0)

    return n_cv
