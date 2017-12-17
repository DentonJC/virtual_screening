#!/usr/bin/env python

import os
import sys
import logging
import argparse
import numpy as np
from datetime import datetime
from sklearn.svm import SVC
from sklearn.utils import class_weight as cw
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer
from src.main import create_callbacks, read_config, evaluate, start_log
from src.data_loader import get_data
from src.models.models import build_logistic_model
from src.models.models import build_residual_model
from keras.wrappers.scikit_learn import KerasClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

xgb_flag = False
try:
    import xgboost as xgb
    xgb_flag = True
except ImportError:
    logger.info("xgboost is not available")

scoring = {'accuracy': 'accuracy', 'MCC': make_scorer(matthews_corrcoef)}
n_physical = 196


def get_options():
    parser = argparse.ArgumentParser(prog="model data section")
    parser.add_argument('select_model', nargs='+', help='name of the model, select from list in README'),
    parser.add_argument('data', nargs='+', help='path to dataset'),
    parser.add_argument('section', nargs='+', help='name of section in config file'),
    parser.add_argument('--features', default='all', choices=['all', 'a', 'fingerprint', 'f', 'physical', 'p'], help='take features: all, fingerptint or physical'),
    parser.add_argument('--output', default=os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/tmp/" + str(datetime.now()) + '/', help='path to output directory'),
    parser.add_argument('--configs', default=os.path.dirname(os.path.realpath(__file__)) + "/configs.ini", help='path to config file'),
    parser.add_argument('--fingerprint', default='morgan', choices=['morgan', 'maccs'], help='maccs (167) or morgan (n)'),
    parser.add_argument('--n_bits', default=256, type=int, help='number of bits in Morgan fingerprint'),
    parser.add_argument('--n_iter', default=6, type=int, help='number of iterations in RandomizedSearchCV'),
    parser.add_argument('--n_jobs', default=1, type=int, help='number of jobs'),
    parser.add_argument('--patience', '-p' , default=100, type=int, help='patience of fit'),
    parser.add_argument('--gridsearch', '-g', action='store_true', default=False, help='use gridsearch'),
    parser.add_argument('--dummy', '-d', action='store_true', default=False, help='use only first 1000 rows of dataset'),
    parser.add_argument('--targets', '-t', default=0, type=int, help='set number of target column'),
    parser.add_argument('--experiments_file', '-e', default='experiments.csv', help='where to write results of experiments')
    return parser

    
def script(args_list, random_state=False, p_rparams=False):
    time_start = datetime.now()
    if len(sys.argv) > 1:
        options = get_options().parse_args()
    else:
        options = get_options().parse_args(args_list)
    if options.targets is not list:
        options.targets = [options.targets]

    callbacks_list = create_callbacks(options.output, options.patience, options.data[0])
    
    # writing to a file
    handler = logging.FileHandler(options.output + 'log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # and to stderr (for stdout `stream=sys.stdout`)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    #logging.basicConfig(filename=options.output+'main.log', level=logging.INFO)
    start_log(logger, options.dummy, options.gridsearch, options.fingerprint, options.n_bits, options.configs, options.data[0], options.section[0])
    n_folds, epochs, rparams, gparams = read_config(options.configs, options.section[0])
    x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles = get_data(logger, options.data[0], options.dummy, options.fingerprint, options.n_bits, options.targets, options.features, random_state)
    
    
    if options.gridsearch and not p_rparams:
        logger.info("GRID SEARCH")  
        
        if options.select_model[0] == "logreg":
            model = RandomizedSearchCV(LogisticRegression(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10, 
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "knn":
            model = RandomizedSearchCV(KNeighborsClassifier(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10,
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "xgb" and xgb_flag:
            model = RandomizedSearchCV(xgb.XGBClassifier(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10, 
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "svc":
            model = RandomizedSearchCV(SVC(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10, 
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "rf":
            model = RandomizedSearchCV(RandomForestClassifier(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10, 
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "if":
            model = RandomizedSearchCV(IsolationForest(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=n_folds, verbose=10, 
                                       scoring=scoring, refit='MCC')
        elif options.select_model[0] == "regression":
            search_model = KerasClassifier(build_fn=build_logistic_model, input_dim=input_shape, output_dim=output_shape)
            grid = RandomizedSearchCV(estimator=search_model, param_distributions=gparams, n_jobs=options.n_jobs, cv=n_folds, n_iter=options.n_iter, verbose=10, scoring=scoring, refit='MCC')
            rparams = grid.fit(x_train, y_train)
            model = build_residual_model(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_0", 'softmax'), activation_2=rparams.get("activation_0", 'softmax'),
                                     loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']),
                                     optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.001),
                                     momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'), dropout=rparams.get("dropout", 0), layers=rparams.get("layers", 0))
            search_model = KerasClassifier(build_fn=model, input_dim=input_shape, output_dim=output_shape)
        elif options.select_model[0] == "residual":
            search_model = KerasClassifier(build_fn=build_residual_model, input_dim=input_shape, output_dim=output_shape)
            grid = RandomizedSearchCV(estimator=search_model, param_distributions=gparams, n_jobs=options.n_jobs, cv=n_folds, n_iter=options.n_iter, verbose=10, scoring=scoring, refit='MCC')
            rparams = grid.fit(x_train, y_train)
            model = build_logistic_model(input_shape, output_shape, activation=rparams.get("activation"),
                                     loss=rparams.get("loss"), metrics=rparams.get("metrics"),
                                     optimizer=rparams.get("optimizer"), learning_rate=rparams.get("learning_rate"),
                                     momentum=rparams.get("momentum"), init_mode=rparams.get("init_mode")) 
        else:
            logger.info("Model name is not found or xgboost import error.")
            return 0, 0

        logger.info("FIT")
        history = model.fit(x_train, np.ravel(y_train))

    else:
        if p_rparams:
            rparams = p_rparams
        if options.select_model[0] == "logreg":
            rparams['verbose'] = 10
            model = LogisticRegression(**rparams)
        elif options.select_model[0] == "knn":
            model = KNeighborsClassifier(**rparams)
        elif options.select_model[0] == "xgb":
            model = xgb.XGBClassifier(**rparams)
        elif options.select_model[0] == "svc":
            rparams['class_weight'] = "balanced"
            model = SVC(**rparams)
        elif options.select_model[0] == "rf":
            rparams['class_weight'] = "balanced"
            model = RandomForestClassifier(**rparams)
        elif options.select_model[0] == "if":
            model = IsolationForest(**rparams)
        elif options.select_model[0] == "regression":
            model = build_residual_model(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_0", 'softmax'), activation_2=rparams.get("activation_0", 'softmax'),
                                     loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']),
                                     optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.001),
                                     momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'), dropout=rparams.get("dropout", 0), layers=rparams.get("layers", 0))
        elif options.select_model[0] == "residual":
            model = build_logistic_model(input_shape, output_shape, activation=rparams.get("activation"),
                                     loss=rparams.get("loss"), metrics=rparams.get("metrics"),
                                     optimizer=rparams.get("optimizer"), learning_rate=rparams.get("learning_rate"),
                                     momentum=rparams.get("momentum"), init_mode=rparams.get("init_mode"))
        else:
            logger.info("Model name is not found or xgboost import error.")
            return 0, 0

        logger.info("FIT")

        if options.select_model[0] == "regression" or options.select_model[0] == "residual":
            if not rparams.get("class_weight"):
                y = [item for sublist in y_train for item in sublist]
                class_weight = cw.compute_class_weight("balanced", np.unique(y), y)
            history = model.fit(x_train, np.ravel(y_train), batch_size=rparams.get("batch_size"), epochs=epochs, validation_data=(x_val, y_val), shuffle=True, verbose=1, callbacks=callbacks_list, class_weight=class_weight)
        else:
            history = model.fit(x_train, np.ravel(y_train))

    if options.gridsearch and not p_rparams:
        rparams = model.cv_results_

    logger.info("EVALUATE")
    train_acc, test_acc, rec = evaluate(logger, options, random_state, options.output, model, x_train, x_test, x_val, y_train, y_test, y_val, time_start, rparams, history, options.data[0], options.section[0], options.features[0], n_jobs=options.n_jobs)
    return train_acc, test_acc, rparams, rec
    

if __name__ == "__main__":
    args_list = ['data/preprocessed/tox21_morgan_256.csv', 'LOGREG_TOX21', '--features', 'p', '--fingerprint', 'morgan', '--n_bits', '256', '--n_jobs', '-1', '-p', '200', '-t', '0']
    script(args_list)
