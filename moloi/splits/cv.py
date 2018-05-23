from moloi.splits.scaffold_split import scaffold_split
from moloi.splits.cluster_split import cluster_split


def create_cv(smiles, split_type, n_cv, random_state):
    if split_type == "scaffold":
        count = n_cv
        n_cv = [([], []) for _ in range(count)]
        for i in range(count):
            train, test = scaffold_split(smiles, frac_train = 1 - ((len(smiles) / count) / (len(smiles)/100))/100, seed=random_state)
            n_cv[i][0].append(train)
            n_cv[i][1].append(test)
    
    if split_type == "cluster":
        count = n_cv        
        n_cv = [([], []) for _ in range(count)]
        for i in range(count):
            train, test = cluster_split(smiles, test_cluster_id=i, n_splits=count)
            n_cv[i][0].append(train)
            n_cv[i][1].append(test)
    return n_cv
