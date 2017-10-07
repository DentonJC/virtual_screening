#!/usr/bin/env python

import logging
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from src.main import read_cmd, read_config, evaluate
from src.data import get_data



DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list, config_path, section, nBits, set_targets, set_features, n_jobs = read_cmd()
n_folds, epochs, rparams, gparams, n_iter, class_weight = read_config(config_path, section)
x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles = get_data(filename, DUMMY, MACCS, Morgan, nBits, set_targets, set_features)

if GRID_SEARCH:
    model = GridSearchCV(xgb.XGBClassifier(**rparams), gparams, scoring='accuracy', cv=n_folds, n_jobs=n_jobs, verbose=10)
else:
    model = xgb.XGBClassifier(**rparams)

print("FIT")
logging.info("FIT")
model.fit(x_train, y_train.reshape(y_train.shape[0],))

if GRID_SEARCH:
    rparams = model.grid_scores_
    print(rparams)

print("EVALUATE")
logging.info("EVALUATE")
evaluate(path, model, x_train, x_test, x_val, y_train, y_test, y_val, tstart, rparams, None)

