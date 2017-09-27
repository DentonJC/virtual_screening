import sys
import os
import getpass
import logging
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", "")) # add path to PATH for src.* imports
from src.main import read_cmd, read_config, evaluate
from src.gridsearch import grid_search
from src.data import get_data
from src.models.models import build_residual_model
from sklearn.utils import class_weight as cw
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list, config_path, section, nBits = read_cmd()
logging.basicConfig(filename=path + 'main.log', level=logging.INFO)

n_folds, epochs, set_targets, set_features, rparams, gparams, n_iter, class_weight = read_config(config_path, section)
    
x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles = get_data(filename, DUMMY, MACCS, Morgan, nBits, set_targets, set_features) 

etc_model = ExtraTreesClassifier(n_jobs=-1) 
etc = etc_model.fit(x_train, y_train)
f_train = etc.transform(x_train)
f_test = etc.transform(x_test)
f_val = etc.transform(x_val)
_, input_shape = f_train.shape

if GRID_SEARCH:
    rparams = grid_search(gparams, build_residual_model, f_train, y_train, input_shape, output_shape, path, n_folds, n_iter)

model = build_residual_model(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_0", 'softmax'), activation_2=rparams.get("activation_0", 'softmax'),
                            loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']), 
                            optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.001), 
                            momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'), dropout=rparams.get("dropout", 0), layers=rparams.get("layers", 0))
print("FIT")
logging.info("FIT")
if not class_weight:
    y = [item for sublist in y_train for item in sublist]
    class_weight = cw.compute_class_weight("balanced", np.unique(y), y)
history = model.fit(f_train, y_train, batch_size=rparams.get("batch_size"), epochs=epochs, validation_data=(f_val, y_val), shuffle=True, verbose=1, callbacks=callbacks_list, class_weight=class_weight)
print("EVALUATE")
logging.info("EVALUATE")
evaluate(path, model, f_train, f_test, f_val, y_train, y_test, y_val, tstart, rparams, history)
