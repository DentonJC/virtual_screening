import sys
import os
import getpass
import logging

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", ""))

from src.main import *
patience = 20
pyname, DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list, config_addr = firing(patience)
logging.basicConfig(filename=path+'residual.log', level=logging.INFO)

exec("from "  + config_addr + " import *")

#from src.configs.config_regression_hiv import *
from src.gridsearch import *
from src.data import get_data_bio_csv
from src.models.models import build_residual_model

seed = 7
np.random.seed(seed)

#filename = os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", "") + "/data/HIV_maccs.csv"
#MACCS = True
#Morgan = False

if MACCS:
    input_shape = 167
if Morgan:
    input_shape = 1024

x_train, x_test, y_train, y_test, output_shape = get_data_bio_csv(filename, input_shape, DUMMY, MACCS, Morgan) 

if GRID_SEARCH:
    rparams = grid_search(gparams, build_residual_model, x_train, y_train, input_shape, output_shape, path, n_folds=10)

model = build_residual_model(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_1", 'softmax'), activation_2=rparams.get("activation_2", 'softmax'),
                            loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']), 
                            optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", '0.001'), 
                            momentum=rparams.get("momentum", '0.1'), init_mode=rparams.get("init_mode", 'uniform'))

print("FIT")
logging.info("FIT")
history = model.fit(x_train, y_train, batch_size=rparams.get("batch_size", 32), epochs=epochs, validation_data=(x_test, y_test), shuffle=True, verbose=1, callbacks=callbacks_list)
print("EVALUATE")
logging.info("EVALUATE")
evaluate_and_done(path, model, x_test, y_test, tstart, rparams, history, pyname)
