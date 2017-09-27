import sys
import os
import getpass
import logging
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", "")) # add path to PATH for src.* imports
from src.main import read_cmd, read_config, evaluate
from src.gridsearch import grid_search
from src.data import get_data
from src.models.models import build_logistic_model
from sklearn.utils import class_weight as cw
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list, config_path, section, nBits = read_cmd()
logging.basicConfig(filename=path + 'main.log', level=logging.INFO)

n_folds, epochs, set_targets, set_features, rparams, gparams, n_iter, class_weight = read_config(config_path, section)

x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles = get_data(filename, DUMMY, MACCS, Morgan, nBits, set_targets, set_features) 

test = SelectKBest(score_func=chi2, k=50)
skb = test.fit(x_train, y_train)
print(skb.scores_)
f_train = skb.transform(x_train)
f_test = skb.transform(x_test)
f_val = skb.transform(x_val)
_, input_shape = f_train.shape

if GRID_SEARCH:
    rparams = grid_search(gparams, build_logistic_model, f_train, y_train, input_shape, output_shape, path, n_folds, n_iter)

model = build_logistic_model(input_shape, output_shape, activation=rparams.get("activation"), 
                            loss=rparams.get("loss"), metrics=rparams.get("metrics"), 
                            optimizer=rparams.get("optimizer"), learning_rate=rparams.get("learning_rate"), 
                            momentum=rparams.get("momentum"), init_mode=rparams.get("init_mode"))    

print("FIT")
logging.info("FIT")

if not class_weight:
    y = [item for sublist in y_train for item in sublist]
    class_weight = cw.compute_class_weight("balanced", np.unique(y), y)
history = model.fit(f_train, y_train, batch_size=rparams.get("batch_size"), epochs=epochs, validation_data=(f_val, y_val), shuffle=True, verbose=1, callbacks=callbacks_list, class_weight=class_weight)
print("EVALUATE")
logging.info("EVALUATE")
evaluate(path, model, f_train, f_test, f_val, y_train, y_test, y_val, tstart, rparams, history)
