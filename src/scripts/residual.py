import sys
import os
import getpass
import logging
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", "")) # add path to PATH for src.* imports
from src.main import read_cmd, read_config, evaluate
from src.gridsearch import grid_search
from src.data import get_data
from src.models.models import build_residual_model


DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list, config_path, section, nBits = read_cmd()
logging.basicConfig(filename=path + 'main.log', level=logging.INFO)

n_folds, epochs, set_targets, set_features, rparams, gparams = read_config(config_path, section)
    
x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles = get_data(filename, DUMMY, MACCS, Morgan, nBits, set_targets, set_features) 

if GRID_SEARCH:
    rparams = grid_search(gparams, build_residual_model, x_train, y_train, input_shape, output_shape, path, n_folds=n_folds)

model = build_residual_model(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_0", 'softmax'), activation_2=rparams.get("activation_0", 'softmax'),
                            loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']), 
                            optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.001), 
                            momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'), dropout=rparams.get("dropout", 0), layers=rparams.get("layers", 0))
print("FIT")
logging.info("FIT")
history = model.fit(x_train, y_train, batch_size=rparams.get("batch_size", 32), epochs=epochs, validation_data=(x_val, y_val), shuffle=True, verbose=1, callbacks=callbacks_list)
print("EVALUATE")
logging.info("EVALUATE")
evaluate(path, model, x_train, x_test, x_val, y_train, y_test, y_val, tstart, rparams, history)



