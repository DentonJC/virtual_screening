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
from sklearn.externals import joblib


def grid_search(param_grid, create_model, x_train, y_train, input_shape, output_shape, path, n_folds):
    
    
    print("GRID SEARCH")
    logging.info("GRID SEARCH")
    #if not os.path.exists(path+"grid/*"):
    #    os.makedirs(path+"grid/")
    #checkpoint = ModelCheckpoint(path+"grid/"+str(datetime.now())+"-{epoch:02d}-{acc:.2f}"+".hdf5", monitor='acc', period=1, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    #csv_logger = CSVLogger(path+'history_grid_'+os.path.basename(sys.argv[0]).replace(".py", "")+'.csv', append=True, separator=';')
    callbacks_list = []

    search_model = KerasClassifier(build_fn=create_model, input_dim = input_shape, output_dim = output_shape)
    orig_stdout = sys.stdout
    f = open(path+'gridsearch.log', 'w')
    sys.stdout = Logger(sys.stdout, f)
    grid = GridSearchCV(estimator=search_model, param_grid=param_grid, n_jobs=-1, cv=n_folds, fit_params=dict(callbacks=callbacks_list), verbose=10)
    grid_result = grid.fit(x_train, y_train)
    sys.stdout = orig_stdout
    #joblib.dump(grid, path+'grid/'+str(datetime.now())+'_output.pkl')
    #joblib.dump(grid_result.best_estimator_, path+'best_params.pkl')
       
    
    
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

class Logger(object):
    """https://stackoverflow.com/questions/11325019/output-on-the-console-and-file-using-python"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()    
