#!/usr/bin/env python

import sys
import logging
import numpy as np
from sklearn.utils import class_weight as cw
from src.main import read_cmd, read_config, evaluate
from src.gridsearch import grid_search
from src.data import get_data
from src.models.models import build_residual_model
from src.report import auc


DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list, config_path, section, nBits, set_targets, set_features, n_jobs = read_cmd()
n_folds, epochs, rparams, gparams, n_iter, class_weight = read_config(config_path, section)
x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles = get_data(filename, DUMMY, MACCS, Morgan, nBits, set_targets, set_features)

if GRID_SEARCH:
    rparams = grid_search(gparams, build_residual_model, x_train, y_train, input_shape, output_shape, path, n_folds, n_iter, n_jobs)

model = build_residual_model(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_0", 'softmax'), activation_2=rparams.get("activation_0", 'softmax'),
                             loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']),
                             optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.001),
                             momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'), dropout=rparams.get("dropout", 0), layers=rparams.get("layers", 0))

print("FIT")
logging.info("FIT")
if not class_weight:
    y = [item for sublist in y_train for item in sublist]
    class_weight = cw.compute_class_weight("balanced", np.unique(y), y)
history = model.fit(x_train, y_train, batch_size=rparams.get("batch_size"), epochs=epochs, validation_data=(x_val, y_val), shuffle=True, verbose=1, callbacks=callbacks_list, class_weight=class_weight)
print("EVALUATE")
logging.info("EVALUATE")
evaluate(path, model, x_train, x_test, x_val, y_train, y_test, y_val, tstart, rparams, history)
auc(model, x_train, x_test, x_val, y_train, y_test, y_val, path)
