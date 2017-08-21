import pickle  
import time  
import json
import sklearn
import sys
import os
import logging
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

def grid_search(param_grid, create_model, x_train, y_train, input_shape, output_shape, path, n_folds):
    print("GRID SEARCH")
    logging.info("GRID SEARCH")
    if not os.path.exists(path+"grid/*"):
        os.makedirs(path+"grid/")
    checkpoint = ModelCheckpoint(path+"grid/"+str(datetime.now())+"-{epoch:02d}-{acc:.2f}"+".hdf5", monitor='acc', period=1, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    csv_logger = CSVLogger(path+'history_grid_'+os.path.basename(sys.argv[0]).replace(".py", "")+'.csv', append=True, separator=';')
    callbacks_list = []

    search_model = KerasClassifier(build_fn=create_model, input_dim = input_shape, output_dim = output_shape)
    grid = GridSearchCV(estimator=search_model, param_grid=param_grid, n_jobs=-1, cv=n_folds, fit_params=dict(callbacks=callbacks_list))
    grid_result = grid.fit(x_train, y_train)
    
    orig_stdout = sys.stdout
    f = open(path+'grid_params', 'w')
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
