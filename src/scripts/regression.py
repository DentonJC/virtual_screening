import sys
import os
import getpass
import logging
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", ""))

from src.main import *

pyname, DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list, config_addr, nBits = firing()

logging.basicConfig(filename=path+'regression.log', level=logging.INFO)

from src.gridsearch import *
from src.data import get_data_bio_csv
from src.models.models import build_logistic_model

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

if GRID_SEARCH:
    rparams = grid_search(gparams, build_logistic_model, x_train, y_train, input_shape, output_shape, path, n_folds=10)

model = build_logistic_model(input_shape, output_shape, activation=rparams.get("activation", 'softmax'), 
                            loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']), 
                            optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", '0.001'), 
                            momentum=rparams.get("momentum", '0.1'), init_mode=rparams.get("init_mode", 'uniform'))    

print("FIT")
logging.info("FIT")
history = model.fit(x_train, y_train, batch_size=rparams.get("batch_size", 32), epochs=epochs, validation_data=(x_val, y_val), shuffle=True, verbose=1, callbacks=callbacks_list)
print("EVALUATE")
logging.info("EVALUATE")
evaluate_and_done(path, model, x_train, x_test, x_val, y_train, y_test, y_val, tstart, rparams, history, pyname)
