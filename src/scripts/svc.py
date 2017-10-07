#!/usr/bin/env python

import logging
import numpy as np
from sklearn.utils import class_weight as cw
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from src.main import read_cmd, read_config, evaluate
from src.data import get_data

DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list, config_path, section, nBits, set_targets, set_features, n_jobs = read_cmd()
n_folds, epochs, rparams, gparams, n_iter, class_weight = read_config(config_path, section)
x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles = get_data(filename, DUMMY, MACCS, Morgan, nBits, set_targets, set_features)


if not class_weight:
    y = [item for sublist in y_train for item in sublist]
    class_w = cw.compute_class_weight("balanced", np.unique(y), y)
    class_weight = {0: class_w[0], 1: class_w[1]}

if GRID_SEARCH:
    model = GridSearchCV(SVC(**rparams), gparams, scoring='accuracy', cv=n_folds, n_jobs=n_jobs, verbose=10)
else:
    model = SVC(**rparams)

print("FIT")
logging.info("FIT")
history = model.fit(x_train, y_train.reshape(y_train.shape[0],))

if GRID_SEARCH:
    rparams = model.grid_scores_
    print(rparams)

print("EVALUATE")
logging.info("EVALUATE")
evaluate(path, model, x_train, x_test, x_val, y_train, y_test, y_val, tstart, rparams, history)
