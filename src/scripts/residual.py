import sys
import os
import getpass
import logging

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", "")) # add path to PATH for src.* imports

from src.main import *

pyname, DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list, config_addr, nBits = firing()
logging.basicConfig(filename=path+'residual.log', level=logging.INFO)

from src.gridsearch import *
from src.data import get_data_bio_csv
from src.models.models import build_residual_model


seed = 7
np.random.seed(seed)

if MACCS:
    input_shape = 167+196
if Morgan:
    input_shape = nBits+196

x_train, x_test, x_val, y_train, y_test, y_val, output_shape, smiles = get_data_bio_csv(filename, input_shape, DUMMY, MACCS, Morgan, nBits) 
exec("from "  + config_addr + " import *")
if targets:
    y_train = y_train[:,targets].reshape(y_train.shape[0],len(targets))
    y_test = y_test[:,targets].reshape(y_test.shape[0],len(targets))
    y_val = y_val[:,targets].reshape(y_val.shape[0],len(targets))
    
if features:
    x_train = x_train[:,features].reshape(x_train.shape[0],len(features))
    x_test = x_test[:,features].reshape(x_test.shape[0],len(features))
    x_val = x_val[:,features].reshape(x_val.shape[0],len(features))

x_train, y_train = drop_nan(x_train, y_train)
x_test, y_test = drop_nan(x_test, y_test)
x_val, y_val = drop_nan(x_val, y_val)
print("X_train:", x_train.shape)
print("Y_train:", y_train.shape)
print("X_test:", x_test.shape)
print("Y_test:", y_test.shape)
print("X_val:", x_val.shape)
print("Y_val:", y_val.shape)
_, input_shape = x_train.shape
_, output_shape = y_train.shape

if GRID_SEARCH:
    rparams = grid_search(gparams, build_residual_model, x_train, y_train, input_shape, output_shape, path, n_folds=10)

print("HERE",input_shape, output_shape)
model = build_residual_model(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_0", 'softmax'), activation_2=rparams.get("activation_0", 'softmax'),
                            loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']), 
                            optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.001), 
                            momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'), dropout=rparams.get("dropout", 0), layers=rparams.get("layers", 0))

print("FIT")
logging.info("FIT")
history = model.fit(x_train, y_train, batch_size=rparams.get("batch_size", 32), epochs=epochs, validation_data=(x_val, y_val), shuffle=True, verbose=1, callbacks=callbacks_list)
print("EVALUATE")
logging.info("EVALUATE")
evaluate_and_done(path, model, x_train, x_test, x_val, y_train, y_test, y_val, tstart, rparams, history, pyname)
