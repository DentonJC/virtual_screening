import sys
import os
import getpass
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", ""))

from src.main import *
from src.configs.config_muv import *
from src.gridsearch import *
from src.data import get_data_bio_csv
from src.models.models import build_residual_model

seed = 7
np.random.seed(seed)

pyname, DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list = firing(patience)
#filename = os.path.dirname(os.path.realpath(__file__)).replace("/src", "").replace("/scripts", "") + "/data/tox21_maccs.csv"
#MACCS = True
#Morgan = False

if MACCS:
    input_shape = 167
if Morgan:
    input_shape = 1024

x_train, x_test, y_train, y_test, output_shape = get_data_bio_csv(filename, input_shape, DUMMY, MACCS, Morgan) 

if GRID_SEARCH:
    rparams = grid_search(gparams, build_residual_model, x_train, y_train, input_shape, output_shape)

model = build_residual_model(input_shape, output_shape)

print("FIT")
history = model.fit(x_train, y_train, batch_size=rparams.get("batch_size", 32), epochs=epochs, validation_data=(x_test, y_test), shuffle=True, verbose=1, callbacks=callbacks_list)
print("EVALUATE")
evaluate_and_done(path, model, x_test, y_test, tstart, rparams, history, pyname)
