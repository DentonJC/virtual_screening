#!/usr/bin/env python

import os
import sys
import time
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
from sklearn.preprocessing import MinMaxScaler, Normalizer
from keras.wrappers.scikit_learn import KerasClassifier
from moloi.config_processing import read_model_config, cv_splits_save, cv_splits_load
from moloi.evaluation import evaluate, make_scoring
from moloi.splits.cv import create_cv
from moloi.data_processing import get_data, clean_data, drop_nan
from moloi.models.keras_models import FCNN, LSTM, MLP, Logreg, create_callbacks
from moloi.model_processing import load_model, save_model
import xgboost as xgb


def get_options():
    parser = argparse.ArgumentParser(prog="model data section")
    parser.add_argument('--select_model', help='name of the model, select from list in README'),
    parser.add_argument('--data_config', help='path to dataset config file'),
    parser.add_argument('--section', help='name of section in model config file'),
    parser.add_argument('--load_model',  help='path to model .sav'),
    parser.add_argument('--descriptors', default=['mordred', 'maccs'], help='descriptor of molecules'),
    parser.add_argument('--output', default=os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp/" + str(datetime.now()) + '/', help='path to output directory'),
    parser.add_argument('--model_config', default="/data/model_configs/bace.ini", help='path to config file'),
    parser.add_argument('--n_bits', default=256, type=int, help='number of bits in Morgan fingerprint'),
    parser.add_argument('--n_cv', default=5, type=int, help='number of splits in RandomizedSearchCV'),
    parser.add_argument('--n_iter', default=6, type=int, help='number of iterations in RandomizedSearchCV'),
    parser.add_argument('--n_jobs', default=-1, type=int, help='number of jobs'),
    parser.add_argument('--patience', '-p' , default=100, type=int, help='patience of fit'),
    parser.add_argument('--gridsearch', '-g', action='store_true', default=False, help='use gridsearch'),
    parser.add_argument('--metric', default='accuracy', choices=['accuracy', 'roc_auc', 'f1', 'matthews'],  help='metric for RandomizedSearchCV'),
    parser.add_argument('--split_type', choices=['stratified', 'scaffold', 'random', 'cluster'], default='stratified', type=str, help='type of train-test split'),
    parser.add_argument('--split_s', default=0.2, type=float, help='size of test and valid splits'),
    parser.add_argument('--targets', '-t', default=0, help='set number of target column'),
    parser.add_argument('--experiments_file', '-e', default='experiments.csv', help='where to write results of experiments')
    return parser

    
def experiment(args_list, random_state=False, p_rparams=False, verbose=0, logger_flag=False):
    time_start = datetime.now()
    # processing parameters
    args_list= list(map(str, args_list))
    if len(sys.argv) > 1:
        options = get_options().parse_args()
    else:
        options = get_options().parse_args(args_list)

    if type(options.targets) is str:
        options.targets = eval(options.targets)
    if type(options.targets) is not list:
        options.targets = [options.targets]

    descriptors = options.descriptors
    options.descriptors = eval(options.descriptors) # input is string, need array
    n_cv = options.n_cv

    # create experiment folder before starting log
    if not os.path.exists(options.output):
        os.makedirs(options.output)
    if not os.path.exists(options.output+"results/*"):
        os.makedirs(options.output+"results/")

    # create logger object, it is passed to all functions in the program
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    # writing log to file
    handler = logging.FileHandler(options.output + 'log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not logger_flag:
        # writing log to terminal (for stdout `stream=sys.stdout`)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info("Script adderss: %s", str(sys.argv[0]))
    logger.info("Descriptors: %s", str(options.descriptors))
    logger.info("n_bits: %s", str(options.n_bits))
    logger.info("Config file: %s", str(options.model_config))
    logger.info("Section: %s", str(options.section))
    if options.gridsearch:
        logger.info("Grid search")
    # load data and configs
    root_address = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
    epochs, rparams, gparams = read_model_config(root_address+options.model_config, options.section)
    x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape, smiles = get_data(logger, root_address+options.data_config, options.n_bits, 
                                                                                                options.targets, random_state, options.split_type, options.split_s, 
                                                                                                verbose, options.descriptors, options.n_jobs)

    x_train = clean_data(x_train)
    x_test = clean_data(x_test)
    x_val = clean_data(x_val)

    # Scale
    if options.select_model == "svc":
        transformer_X = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
    else:
        transformer_X = MinMaxScaler().fit(x_train)
    x_train = transformer_X.transform(x_train)
    x_test = transformer_X.transform(x_test)
    x_val = transformer_X.transform(x_val)

    if len(np.unique(y_train)) == 1:
        logger.error("Multiclass data: only one class in y_train")
        sys.exit(0)
    if len(np.unique(y_test)) == 1:
        logger.error("Multiclass data: only one class in y_test")
        sys.exit(0)
    if len(np.unique(y_val)) == 1:
        logger.error("Multiclass data: only one class in y_val")
        sys.exit(0)
    if (len(np.unique(y_train)) > 2 or len(np.unique(y_test)) > 2 or len(np.unique(y_val)) > 2) and "roc_auc" in options.metric:
        logger.error("Multiclass data: can not use ROC AUC metric")
        sys.exit(0)
    if y_train.shape[1] > 1 and "roc_auc" in options.metric:
        logger.error("Multilabel data: can not use ROC AUC metric")
        sys.exit(0)

    history = False
    model_loaded = False
    grid = False
        
    if options.load_model:
        _, model_loaded = load_model(options.load_model, logger)

    if options.gridsearch and not p_rparams and not model_loaded:
        logger.info("GRID SEARCH")
        logger.info("x_train shape: %s", str(np.array(x_train).shape))
        logger.info("x_test shape: %s", str(np.array(x_test).shape))
        logger.info("x_val shape: %s", str(np.array(x_val).shape))
        logger.info("y_train shape: %s", str(np.array(y_train).shape))
        logger.info("y_test shape: %s", str(np.array(y_test).shape))
        logger.info("y_val shape: %s", str(np.array(y_val).shape))
        #scoring = {'accuracy': 'accuracy', 'MCC': make_scorer(matthews_corrcoef)}
        scoring = make_scoring(options.metric)
        ####
        loaded_cv = cv_splits_load(options.split_type, options.split_s, root_address+options.data_config, options.targets)
        if loaded_cv is False:
            for i in range(100):
                count = 0
                options.n_cv = create_cv(smiles, options.split_type, options.n_cv, random_state, y_train)
                for j in options.n_cv:
                    if len(np.unique(j)) > 1:
                        count += 1
                if count == len(options.n_cv):
                    break
                random_state += 1
            if count != len(options.n_cv):
                logger.info("Can not create a good split cv. Try another random_seed or check the dataset.")
                sys.exit(0)

        else:
            options.n_cv = eval(loaded_cv)

        cv_splits_save(options.split_type, options.split_s,  options.n_cv, root_address+options.data_config, options.targets)
        f = open(options.output+'n_cv', 'w')
        f.write(str(options.n_cv))
        f.close()
        ####
        # check if number of iterations more then possible combinations
        try:
            keys = list(gparams.keys())
            n_iter = 1
            for k in keys:
                n_iter *= len(gparams[k])
            if options.n_iter > n_iter:
                options.n_iter = n_iter
        except:
            pass

        sklearn_params = {'param_distributions': gparams,
                         'n_iter': options.n_iter,
                         'n_jobs': options.n_jobs,
                         'cv': options.n_cv,
                         'verbose': verbose,
                         'scoring': scoring,
                         'return_train_score': True,
                         'refit': True,
                         'random_state': random_state}

        keras_params = {'param_distributions': gparams,
                            'n_jobs': options.n_jobs,
                            'cv': options.n_cv,
                            'n_iter': options.n_iter,
                            'verbose': verbose,
                            'scoring': scoring,
                            'random_state': random_state,
                            'return_train_score': True,
                            'refit': True,
                            'pre_dispatch': n_cv}
        
        grid_sklearn_params = {'param_grid': gparams,
                         'n_jobs': options.n_jobs,
                         'cv': options.n_cv,
                         'verbose': verbose,
                         'scoring': scoring,
                         'return_train_score': True,
                         'refit': True}
                         
        # sklearn models
        if options.select_model == "lr":
            model = RandomizedSearchCV(LogisticRegression(**rparams), **sklearn_params)
        elif options.select_model == "knn":
            model = RandomizedSearchCV(KNeighborsClassifier(**rparams), **sklearn_params)
        elif options.select_model == "xgb":
            model = RandomizedSearchCV(xgb.XGBClassifier(**rparams), **sklearn_params)
        elif options.select_model == "svc":
            if type(gparams) == list:
                model = GridSearchCV(SVC(**rparams, probability=True), **grid_sklearn_params)
            else:
                model = RandomizedSearchCV(SVC(**rparams, probability=True), **sklearn_params)
        elif options.select_model == "rf":
            model = RandomizedSearchCV(RandomForestClassifier(**rparams), **sklearn_params)
        elif options.select_model == "if":
            model = RandomizedSearchCV(IsolationForest(**rparams), **sklearn_params)
        # keras models
        elif options.select_model == "regression":
            search_model = KerasClassifier(build_fn=Logreg, input_shape=input_shape, output_shape=output_shape)
            model = RandomizedSearchCV(estimator=search_model, **keras_params)
        elif options.select_model == "fcnn":
            search_model = KerasClassifier(build_fn=FCNN, input_shape=input_shape, output_shape=output_shape)
            model = RandomizedSearchCV(estimator=search_model, **keras_params)                                    
        elif options.select_model == "lstm":
            search_model = KerasClassifier(build_fn=LSTM, input_shape=input_shape, output_shape=output_shape, input_length=x_train.shape[1])
            model = RandomizedSearchCV(estimator=search_model, **keras_params)
        elif options.select_model == "mlp":
            search_model = KerasClassifier(build_fn=MLP, input_shape=input_shape, output_shape=output_shape)
            model = RandomizedSearchCV(estimator=search_model, **keras_params)   
        # elif options.select_model == "rnn":
        #     search_model = KerasClassifier(build_fn=RNN, input_shape=input_shape, output_shape=output_shape, input_length=x_train.shape[1])
        #     model = RandomizedSearchCV(estimator=search_model, **keras_params)        
        # elif options.select_model == "gru":
        #     search_model = KerasClassifier(build_fn=GRU, input_shape=input_shape, output_shape=output_shape, input_length=x_train.shape[1])
        #     model = RandomizedSearchCV(estimator=search_model, **keras_params)
        else:
            logger.info("Model name is not found.")
            sys.exit(0)
        
        time.sleep(5)
        logger.info("GRIDSEARCH FIT")
        if y_train.shape[1] == 1:
            model.fit(x_train, np.ravel(y_train))
        else:
            model.fit(x_train, y_train)
        rparams = model.best_params_
        grid = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending=False)
        grid.to_csv(options.output + "gridsearch.csv")
        # model = model.best_estimator_

    if p_rparams:
        rparams = p_rparams

    r_batch_size = rparams.get("batch_size")
    r_epochs = rparams.get("epochs")
    r_class_weight = rparams.get("class_weight")

    try:
        del rparams["epochs"]
    except KeyError:
        pass
    try:
        del rparams["class_weight"]
    except KeyError:
        pass
    try:
        del rparams["batch_size"]
    except KeyError:
        pass

    if options.select_model == "lr":
        model = LogisticRegression(**rparams)
    elif options.select_model == "knn":
        model = KNeighborsClassifier(**rparams)
    elif options.select_model == "xgb":
        model = xgb.XGBClassifier(**rparams)
    elif options.select_model == "svc":
        model = SVC(**rparams, probability=True)
    elif options.select_model == "rf":
        model = RandomForestClassifier(**rparams)
    elif options.select_model == "if":
        model = IsolationForest(**rparams)

    elif options.select_model == "fcnn":
        model = FCNN(input_shape, output_shape, **rparams)
    elif options.select_model == "regression":
        model = Logreg(input_shape, output_shape, **rparams)
    elif options.select_model == "mlp":
        model = MLP(input_shape, output_shape, **rparams)
    # elif options.select_model == "rnn":
    #     model = RNN(input_shape, output_shape, **rparams)
    # elif options.select_model == "gru":
    #     model = GRU(input_shape, output_shape, **rparams)
    elif options.select_model == "lstm":
       model = LSTM(input_shape, output_shape, **rparams)
    else:
        logger.info("Model name is not found or xgboost import error.")
        sys.exit(0)
    
    rparams["batch_size"] = r_batch_size
    rparams["class_weight"] = r_class_weight
    rparams["epochs"] = r_epochs
    
    if options.load_model:
        model, model_loaded = load_model(options.load_model, logger)
    logger.info("MODEL FIT")
    try:
        monitor = 'binary_crossentropy'
        mode = 'auto'
        #if monitor is False:
        #    monitor = 'val_acc'
        #    mode = 'max'
        callbacks = create_callbacks(options.output, options.patience, options.section, monitor="val_" + monitor, mode=mode)
        history = model.fit(x_train, np.ravel(y_train), batch_size=rparams.get("batch_size"), epochs=epochs, shuffle=False, verbose=verbose, callbacks=callbacks, validation_data=(x_val, y_val))
    except:
        model.fit(x_train, np.ravel(y_train))
    
    logger.info("EVALUATE")
    accuracy_test, accuracy_train, rec, auc, auc_val, f1, path = evaluate(logger, options, random_state, options.output, model, x_train, 
                                                                    x_test, x_val, y_val, y_train, y_test, time_start, rparams, history, 
                                                                    options.section, options.n_jobs, descriptors, grid)
    # transfer learning in row
    #if not options.load_model:
    model_address = save_model(model, path, logger)
    #else:
    #model_address = options.load_model

    logger.info("Done")
    return accuracy_test, accuracy_train, rec, auc, auc_val, f1, rparams, model_address


if __name__ == "__main__":
    args_list = ['lr', '/data/data_configs/bace.ini']
    script(args_list)
