import pickle  
import time  
import json
import sklearn
import sys
import os
import logging
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.externals import joblib
from src.main import Logger


def grid_search(param_grid, create_model, x_train, y_train, input_shape, output_shape, path, n_folds, n_iter, n_jobs):
    print("GRID SEARCH")
    logging.info("GRID SEARCH")
    #callbacks_list = []
    search_model = KerasClassifier(build_fn=create_model, input_dim=input_shape, output_dim=output_shape)
    orig_stdout = sys.stdout
    f = open(path + 'gridsearch.log', 'w')
    sys.stdout = Logger(sys.stdout, f)
    grid = RandomizedSearchCV(estimator=search_model, param_distributions=param_grid, n_jobs=n_jobs, cv=n_folds, n_iter=n_iter, verbose=10) # fit_params=dict(callbacks=callbacks_list))
    grid_result = grid.fit(x_train, y_train)
    sys.stdout = orig_stdout
    f.close()
    
    f = open(path + 'grid_params', 'w')
    sys.stdout = f
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        logging.info("%f (%f) with: %r" % (mean, stdev, param))
    sys.stdout = orig_stdout
    f.close()
    
    return grid_result.best_params_
