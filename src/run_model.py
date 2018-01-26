#!/usr/bin/env python

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVC
from sklearn.utils import class_weight as cw
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit
from src.main import read_model_config, evaluate, start_log, make_scoring
from src.data_loader import get_data
from src.report_formatter import plot_grid_search
from src.models.keras_models import create_callbacks
from src.models.keras_models import Perceptron, Residual, LSTM, MLSTM, RNN, MRNN, GRU
from keras.wrappers.scikit_learn import KerasClassifier
import xgboost as xgb


def get_options():
    parser = argparse.ArgumentParser(prog="model data section")
    parser.add_argument('select_model', nargs='+', help='name of the model, select from list in README'),
    parser.add_argument('data_config', nargs='+', help='path to dataset config file'),
    parser.add_argument('section', nargs='+', help='name of section in model config file'),
    parser.add_argument('--load_model',  help='path to model .sav'),
    parser.add_argument('--features', default='all', choices=['all', 'a', 'fingerprint', 'f', 'physical', 'p'], help='take features: all, fingerptint or physical'),
    parser.add_argument('--output', default=os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/tmp/" + str(datetime.now()) + '/', help='path to output directory'),
    parser.add_argument('--model_config', default="/data/model_configs/bace.ini", help='path to config file'),
    parser.add_argument('--fingerprint', default='morgan', choices=['morgan', 'maccs'], help='maccs (167) or morgan (n)'),
    parser.add_argument('--n_bits', default=256, type=int, help='number of bits in Morgan fingerprint'),
    parser.add_argument('--n_cv', default=5, type=int, help='number of splits in RandomizedSearchCV'),
    parser.add_argument('--n_iter', default=6, type=int, help='number of iterations in RandomizedSearchCV'),
    parser.add_argument('--n_jobs', default=1, type=int, help='number of jobs'),
    parser.add_argument('--patience', '-p' , default=100, type=int, help='patience of fit'),
    parser.add_argument('--gridsearch', '-g', action='store_true', default=False, help='use gridsearch'),
    parser.add_argument('--metric', default='accuracy', choices=['accuracy', 'roc_auc', 'f1', 'matthews'],  help='metric for RandomizedSearchCV'),
    parser.add_argument('--split', default=0.2, type=float, help='train-test split'),
    parser.add_argument('--targets', '-t', default=0, type=int, help='set number of target column'),
    parser.add_argument('--experiments_file', '-e', default='experiments.csv', help='where to write results of experiments')
    return parser

    
def script(args_list, random_state=False, p_rparams=False, verbose=0):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    args_list= list(map(str, args_list))
    print(args_list)
    time_start = datetime.now()
    if len(sys.argv) > 1:
        options = get_options().parse_args()
    else:
        options = get_options().parse_args(args_list)
    if options.targets is not list:
        options.targets = [options.targets]
    callbacks = create_callbacks(options.output, options.patience, options.section[0])
    
    # writing to a file
    handler = logging.FileHandler(options.output + 'log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # and to stderr (for stdout `stream=sys.stdout`)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    #logging.basicConfig(filename=options.output+'main.log', level=logging.INFO)
    start_log(logger, options.gridsearch, options.fingerprint, options.n_bits, options.model_config, options.section[0])
    epochs, rparams, gparams = read_model_config(os.path.dirname(os.path.realpath(__file__)).replace("/src", "")+options.model_config, options.section[0])
    x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape = get_data(logger, os.path.dirname(os.path.realpath(__file__)).replace("/src", "")+options.data_config[0], options.fingerprint, options.n_bits, options.targets, options.features, random_state, options.split, verbose)
    
    #scoring = {'accuracy': 'accuracy', 'MCC': make_scorer(matthews_corrcoef)}
    scoring = make_scoring(options.metric)
    
    if options.gridsearch and not p_rparams:
        logger.info("GRID SEARCH")  
        
        # check if the number of iterations more then possible combinations
        keys = list(gparams.keys())
        n_iter = 1
        for k in keys:
            n_iter*=len(gparams[k])
        if options.n_iter > n_iter: options.n_iter = n_iter
        
        if options.load_model:
            model = pickle.load(open(options.load_model, 'rb'))
        elif options.select_model[0] == "logreg":
            model = RandomizedSearchCV(LogisticRegression(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=options.n_cv, verbose=verbose, 
                                       scoring=scoring, random_state=random_state)
        elif options.select_model[0] == "knn":
            model = RandomizedSearchCV(KNeighborsClassifier(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=options.n_cv, verbose=verbose,
                                       scoring=scoring, random_state=random_state)
        elif options.select_model[0] == "xgb" and xgb_flag:
            model = RandomizedSearchCV(xgb.XGBClassifier(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=options.n_cv, verbose=verbose, 
                                       scoring=scoring, random_state=random_state)
        elif options.select_model[0] == "svc":
            model = RandomizedSearchCV(SVC(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=options.n_cv, verbose=verbose, 
                                       scoring=scoring, random_state=random_state)
        elif options.select_model[0] == "rf":
            model = RandomizedSearchCV(RandomForestClassifier(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=options.n_cv, verbose=verbose, 
                                       scoring=scoring, random_state=random_state)
        elif options.select_model[0] == "if":
            model = RandomizedSearchCV(IsolationForest(**rparams), gparams, n_iter=options.n_iter, n_jobs=options.n_jobs, cv=options.n_cv, verbose=verbose, 
                                       scoring=scoring, random_state=random_state)
                                       
                                       
        elif options.select_model[0] == "regression":
            if 'roc_auc' in options.metric:
                search_model = KerasClassifier(build_fn=Perceptron, input_shape=input_shape, output_shape=output_shape)
            else:
                search_model = KerasClassifier(build_fn=Perceptron, input_shape=input_shape, output_shape=output_shape)

            model = RandomizedSearchCV(
                                    estimator=search_model, 
                                    param_distributions=gparams, 
                                    n_jobs=options.n_jobs, 
                                    cv=options.n_cv, 
                                    n_iter=options.n_iter, 
                                    verbose=verbose,
                                    scoring=scoring, 
                                    random_state=random_state
                                    )

        elif options.select_model[0] == "residual":
            search_model = KerasClassifier(build_fn=Residual, input_shape=input_shape, output_shape=output_shape)
            model = RandomizedSearchCV(
                                    estimator=search_model, 
                                    param_distributions=gparams, 
                                    n_jobs=options.n_jobs, 
                                    cv=options.n_cv, 
                                    n_iter=options.n_iter, 
                                    verbose=verbose, 
                                    scoring=scoring, 
                                    random_state=random_state
                                    )
                                    
        elif options.select_model[0] == "lstm":
            search_model = KerasClassifier(build_fn=LSTM, input_shape=input_shape, output_shape=output_shape, input_length=x_train.shape[1])
            model = RandomizedSearchCV(
                                    estimator=search_model, 
                                    param_distributions=gparams, 
                                    n_jobs=options.n_jobs, 
                                    cv=options.n_cv, 
                                    n_iter=options.n_iter, 
                                    verbose=verbose, 
                                    scoring=scoring, 
                                    random_state=random_state
                                    )
                                    
        elif options.select_model[0] == "rnn":
            search_model = KerasClassifier(build_fn=RNN, input_shape=input_shape, output_shape=output_shape, input_length=x_train.shape[1])
            model = RandomizedSearchCV(
                                    estimator=search_model, 
                                    param_distributions=gparams, 
                                    n_jobs=options.n_jobs, 
                                    cv=options.n_cv, 
                                    n_iter=options.n_iter, 
                                    verbose=verbose, 
                                    scoring=scoring, 
                                    random_state=random_state
                                    )
        
        elif options.select_model[0] == "gru":
            search_model = KerasClassifier(build_fn=GRU, input_shape=input_shape, output_shape=output_shape, input_length=x_train.shape[1])
            model = RandomizedSearchCV(
                                    estimator=search_model, 
                                    param_distributions=gparams, 
                                    n_jobs=options.n_jobs, 
                                    cv=options.n_cv, 
                                    n_iter=options.n_iter, 
                                    verbose=verbose, 
                                    scoring=scoring, 
                                    random_state=random_state
                                    )

        elif options.select_model[0] == "mlstm":
            batch_size = 64 # tune it

            #for stateful LSTMs, need fixed size batches
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
            x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
            
            num_batches_train = int(x_train.shape[0]/batch_size)
            x_train = x_train[0:num_batches_train*batch_size,:,:]
            y_train = y_train[0:num_batches_train*batch_size]

            num_batches_test = int(x_test.shape[0]/batch_size)
            x_test = x_test[0:num_batches_test*batch_size,:,:]
            y_test = y_test[0:num_batches_test*batch_size]
            
            num_batches_val = int(x_val.shape[0]/batch_size)
            x_val = x_val[0:num_batches_val*batch_size,:,:]
            y_val = y_val[0:num_batches_val*batch_size]
    
            search_model = KerasClassifier(build_fn=MLSTM, input_shape=x_train.shape[2], output_shape=output_shape, batch_size=batch_size)
            model = RandomizedSearchCV(
                                    estimator=search_model, 
                                    param_distributions=gparams, 
                                    n_jobs=options.n_jobs, 
                                    n_iter=options.n_iter, 
                                    verbose=verbose, 
                                    scoring=scoring, 
                                    random_state=random_state
                                    )


        elif options.select_model[0] == "mrnn":
            batch_size = 64 # tune it

            #for stateful LSTMs, need fixed size batches
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
            x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
            
            num_batches_train = int(x_train.shape[0]/batch_size)
            x_train = x_train[0:num_batches_train*batch_size,:,:]
            y_train = y_train[0:num_batches_train*batch_size]

            num_batches_test = int(x_test.shape[0]/batch_size)
            x_test = x_test[0:num_batches_test*batch_size,:,:]
            y_test = y_test[0:num_batches_test*batch_size]
            
            num_batches_val = int(x_val.shape[0]/batch_size)
            x_val = x_val[0:num_batches_val*batch_size,:,:]
            y_val = y_val[0:num_batches_val*batch_size]
    
            search_model = KerasClassifier(build_fn=MRNN, input_shape=input_shape, output_shape=output_shape, batch_size=batch_size)
            model = RandomizedSearchCV(
                                    estimator=search_model, 
                                    param_distributions=gparams, 
                                    n_jobs=options.n_jobs,
                                    n_iter=options.n_iter, 
                                    verbose=verbose, 
                                    scoring=scoring, 
                                    random_state=random_state
                                    )
                                                     
        else:
            logger.info("Model name is not found.")
            return 0, 0, 0, 0, 0, 0

        logger.info("FIT")
        try:
            history = model.fit(x_train, np.ravel(y_train), callbacks=callbacks, validation_data=(x_val, np.ravel(y_val)))
        except TypeError:
            history = model.fit(x_train, np.ravel(y_train))
    logger.info("HERE")
    if not options.gridsearch:
        score = False
        logger.info("HERE INSIDE")
        if p_rparams:
            rparams = p_rparams
        if options.load_model:
            model = pickle.load(open(options.load_model, 'rb'))
        elif options.select_model[0] == "logreg":
            model = LogisticRegression(**rparams)
        elif options.select_model[0] == "knn":
            model = KNeighborsClassifier(**rparams)
        elif options.select_model[0] == "xgb":
            model = xgb.XGBClassifier(**rparams)
        elif options.select_model[0] == "svc":
            model = SVC(**rparams)
        elif options.select_model[0] == "rf":
            model = RandomForestClassifier(**rparams)
        elif options.select_model[0] == "if":
            model = IsolationForest(**rparams)

        elif options.select_model[0] == "residual":
            model = Residual(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_0", 'softmax'), activation_2=rparams.get("activation_0", 'softmax'),
                                     loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['binary_accuracy']),
                                     optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.01),
                                     momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'), dropout=rparams.get("dropout", 0.2), layers=rparams.get("layers", 3))
        elif options.select_model[0] == "regression":
            model = Perceptron(input_shape, output_shape, activation=rparams.get("activation"),
                                     loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['binary_accuracy']),
                                     optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.001),
                                     momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'))
        elif options.select_model[0] == "rnn":
            model = RNN(input_shape, output_shape, activation=rparams.get("activation"),
                                     loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']),
                                     optimizer=rparams.get("optimizer", 'Adam'), layers=rparams.get("layers", 0), neurons_1=rparams.get("neurons_1", 256), neurons_2=rparams.get("neurons_2", 512), embedding_length=rparams.get("embedding_length", 32))
        elif options.select_model[0] == "gru":
            model = GRU(input_shape, output_shape, activation=rparams.get("activation"),
                                     loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']),
                                     optimizer=rparams.get("optimizer", 'Adam'), layers=rparams.get("layers", 0), neurons_1=rparams.get("neurons_1", 256), neurons_2=rparams.get("neurons_2", 512), embedding_length=rparams.get("embedding_length", 32))
        elif options.select_model[0] == "lstm":
            model = LSTM(input_shape, output_shape, activation=rparams.get("activation"),
                                     loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']),
                                     optimizer=rparams.get("optimizer", 'Adam'), neurons_1=rparams.get("neurons_1", 256), 
                                     neurons_2=rparams.get("neurons_2", 512), embedding_length=rparams.get("embedding_length", 32), batch_size=rparams.get("batch_size",32))
        else:
            logger.info("Model name is not found or xgboost import error.")
            return 0, 0

        logger.info("FIT")

        if options.select_model[0] == "regression" or options.select_model[0] == "residual":
            if not rparams.get("class_weight"):
                y = [item for sublist in y_train for item in sublist]
                class_weight = cw.compute_class_weight("balanced", np.unique(y), y)
            history = model.fit(x_train, np.ravel(y_train), batch_size=rparams.get("batch_size"), epochs=epochs, shuffle=True, verbose=verbose, callbacks=callbacks, class_weight=class_weight)
        else:
            history = model.fit(x_train, np.ravel(y_train))
    
    logger.info("SAVE GRIDSEARCH RESULTS")
    if options.gridsearch and not p_rparams:
        rparams = model.best_params_

    if options.gridsearch:
        try:
            grid = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending = False)
            grid.to_csv(options.output + "gridsearch.csv")
        except AttributeError:
            logger.info("Can not save RandomizedSearchCV results")
        
        #try:
        score = pd.DataFrame(model.cv_results_)
        #except:
        #    pass

    logger.info("EVALUATE")
    accuracy_test, accuracy_train, rec, auc, auc_val, f1 = evaluate(logger, options, random_state, options.output, model, x_train, 
                                                                    x_test, x_val, y_val, y_train, y_test, time_start, rparams, history, 
                                                                    options.section[0], options.features[0], options.n_jobs, score)
    return accuracy_test, accuracy_train, rec, auc, auc_val, f1, rparams
    

if __name__ == "__main__":
    args_list = ['logreg', '/data/data_configs/bace.ini']
    script(args_list)
