import sys
import numpy as np
from moloi.splits.scaffold_split import scaffold_split
from moloi.splits.cluster_split import cluster_split
from moloi.config_processing import cv_splits_save, cv_splits_load
from sklearn.model_selection import train_test_split


def create_cv(smiles, split_type, n_cv, random_state, y=False):
    if split_type == "scaffold":
        count = n_cv
        n_cv = [([], []) for _ in range(count)]
        for i in range(count):
            train, test = scaffold_split(smiles, frac_train=1 - ((len(smiles)/count)/(len(smiles)/100))/100,
                                         seed=random_state)
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
            train, test = train_test_split(idx, test_size=1 - ((len(smiles)/count)/(len(smiles)/100))/100,
                                           stratify=stratify, shuffle=shuffle)
            n_cv[i][0].append(train)
            n_cv[i][1].append(test)

    elif split_type == "stratified":
        count = n_cv
        n_cv = [([], []) for _ in range(count)]
        for i in range(count):
            shuffle = True
            stratify = y
            idx = [i for i in range(len(smiles))]
            train, test = train_test_split(idx, test_size=1 - ((len(smiles)/count)/(len(smiles)/100))/100,
                                           stratify=stratify, shuffle=shuffle)
            n_cv[i][0].append(train)
            n_cv[i][1].append(test)

    elif split_type == "generator":
        count = n_cv
        n_cv = [([], []) for _ in range(count)]
        for i in range(count):
            shuffle = True
            stratify = None
            idx = [i for i in range(len(smiles))]
            train, test = train_test_split(idx, test_size=1 - ((len(smiles)/count)/(len(smiles)/100))/100,
                                           stratify=stratify, shuffle=shuffle)
            yield(train, test)

    else:
        print("Wrong split type")
        sys.exit(0)

    return n_cv


def create_cv_splits(logger, options, data, exp_settings, n_cv):
    loaded_cv = cv_splits_load(options.split_type, options.split_s,
                               options.data_config, options.targets)
    if loaded_cv is False:
        for _ in range(100):
            count1 = 0
            count2 = 0
            options.n_cv = create_cv(data["smiles"], options.split_type, options.n_cv,
                                     exp_settings["random_state"], data["y_train"])
            for j in options.n_cv:
                j_train = np.array(j[0])
                j_test = np.array(j[1])
                if len(np.unique(data["y_train"][j_train])) > 1:
                    count1 += 1
                if len(np.unique(data["y_train"][j_test])) > 1:
                    count2 += 1
            if count1 == len(options.n_cv) and count2 == len(options.n_cv):
                break
            else:
                options.n_cv = n_cv
                exp_settings["random_state"] += 1
        if count1 != len(options.n_cv) or count2 != len(options.n_cv):
            logger.info("Can not create a good split cv. Try another random_seed or check the dataset.")
            sys.exit(0)
    else:
        options.n_cv = loaded_cv

    cv_splits_save(options.split_type, options.split_s,  options.n_cv,
                   options.data_config, options.targets)
    f = open(options.output+'n_cv', 'w')
    f.write(str(options.n_cv))
    f.close()

    ####
