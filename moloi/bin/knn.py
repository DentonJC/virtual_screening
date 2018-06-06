#!/usr/bin/env python

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from moloi.config_processing import read_model_config
from moloi.evaluation import evaluate, make_scoring
from moloi.splits.cv import create_cv
from moloi.data_processing import get_data, clean_data


root_address = os.path.dirname(os.path.realpath(__file__)).replace("/moloi/bin", "")
output = root_address + "/tmp/" + str(datetime.now()) + '/'
data_config = "/data/data_configs/bace.ini"
model_config = "/data/model_configs/configs.ini"
section = 'KNN'
descriptors = ['mordred', 'maccs']
n_bits = 256
n_cv = 5
n_iter = 5
n_jobs = -1
patience = 100
metric = 'roc_auc'
split_type = 'scaffold'
split_s = 0.1
targets = [0]
random_state = 1337
verbose = 10

time_start = datetime.now()

# create experiment folder before starting log
if not os.path.exists(output):
    os.makedirs(output)
if not os.path.exists(output+"results/*"):
    os.makedirs(output+"results/")

# create logger object, it is passed to all functions in the program
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

# writing log to file
handler = logging.FileHandler(output + 'log')
handler.setFormatter(formatter)
logger.addHandler(handler)

# writing log to terminal (for stdout `stream=sys.stdout`)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# load data and configs
epochs, rparams, gparams = read_model_config(root_address+model_config, section)
x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape, smiles = get_data(logger, root_address+data_config, n_bits,
                                                                                             targets, random_state, split_type, split_s,
                                                                                             verbose, descriptors, n_jobs)

x_train = clean_data(x_train)
x_test = clean_data(x_test)
x_val = clean_data(x_val)
    
# Scale
transformer_X = MinMaxScaler().fit(x_train)
x_train = transformer_X.transform(x_train)
x_test = transformer_X.transform(x_test)
x_val = transformer_X.transform(x_val)

if len(np.unique(y_train)) == 1 or len(np.unique(y_test)) == 1 or len(np.unique(y_val)) == 1:
    logger.error("Only one class in data")
    sys.exit(0)
if len(np.unique(y_train)) > 2 or len(np.unique(y_test)) > 2 or len(np.unique(y_val)) > 2 and "roc_auc" in metric:
    logger.error("Multiclass data: can not use ROC AUC metric")
    sys.exit(0)

scoring = make_scoring(metric)

n_cv = create_cv(smiles, split_type, n_cv, random_state)

sklearn_params = {'param_distributions': gparams,
                  'n_iter': n_iter,
                  'n_jobs': n_jobs,
                  'cv': n_cv,
                  'verbose': verbose,
                  'scoring': scoring,
                  'return_train_score': True,
                  'random_state': random_state}

model = RandomizedSearchCV(KNeighborsClassifier(**rparams), **sklearn_params)
model.fit(x_train, np.ravel(y_train))
rparams = model.best_params_
grid = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending=False)
grid.to_csv(output + "gridsearch.csv")
model = model.best_estimator_
accuracy_test, accuracy_train, rec, auc, auc_val, f1, path = evaluate(logger, False, random_state, output, model, x_train, x_test, x_val, y_val, y_train, y_test, 
                                                                      time_start, rparams, False, section, n_jobs, descriptors, grid)

print("accuracy_test:", accuracy_test)
print("accuracy_train:", accuracy_train)
print("rec:", str(rec))
print("auc:", str(auc))
print("auc_val:", str(auc_val))
print("f1", f1)
print("Report address:", path)
print("Done")
