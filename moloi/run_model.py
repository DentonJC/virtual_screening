#!/usr/bin/env python

import os
import sys
import time
import pickle
from sklearn.externals import joblib
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
from dask_ml.model_selection import RandomizedSearchCV as dRandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json
from moloi.main import read_model_config, cv_splits_save, cv_splits_load, evaluate, start_log, make_scoring
from moloi.data_loader import get_data
from moloi.report_formatter import plot_grid_search
from moloi.models.keras_models import Perceptron, Residual, LSTM, MLSTM, RNN, MRNN, GRU, MultilayerPerceptron, Logreg, create_callbacks
from moloi.splits.scaffold_split import scaffold_split
from moloi.splits.cluster_split import cluster_split
import xgboost as xgb


def load_model(load_model, logger):
    model_loaded, model = False, False
    if True:
    #try:
        if type(eval(load_model)) is int or type(eval(load_model)) is float:
            fp = open(os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp/addresses")
            for i, line in enumerate(fp):
                if i == eval(load_model) - 1:
                    load_model = eval(line)
                    break
            fp.close()
        else:
            load_model = eval(load_model)
        if load_model[1] == False:
            #model = pickle.load(open(options.load_model, 'rb'))
            model = joblib.load(load_model[0])
        else:
            # load json and create model
            json_file = open(load_model[0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(load_model[1])
            compile_params = [line.rstrip('\n') for line in open(load_model[2])]
            model.compile(loss=compile_params[0], metrics=eval(compile_params[1]), optimizer=compile_params[2])
        model_loaded = True
        logger.info("Model loaded")
    #except:
    #    model_loaded = False
    #    logger.info("Model not loaded")
    return model, model_loaded

    
def save_model(model, path, logger):
    model_address = False
    try:
        # pickle.dump(model, open(options.output+"model.sav", 'wb'))
        joblib.dump(model, path+"model.sav")
        model_address = [path+"model.sav", False, False]
        f = open(path+'addresses', 'w')
        f.write(str(model_address))
        f.close()
        
        f = open(os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp/addresses", 'a')
        f.write(str(model_address)+'\n')
        f.close()
        model_address = sum(1 for line in open(os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp/addresses"))

    except TypeError:
        try:
            # serialize model to JSON
            model_json = model.to_json()
            with open(path+"model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(path+"model.h5")

            loss=rparams.get("loss", 'binary_crossentropy')
            metrics=rparams.get("metrics", ['accuracy'])
            optimizer=rparams.get("optimizer", 'Adam')
            f = open(path+'compile', 'w')
            f.write(loss+'\n'+str(metrics)+'\n'+optimizer)
            f.close()
            model_address = [path+"model.json", path+"model.h5", path+'compile']
            f = open(path+'addresses', 'w')
            f.write(str(model_address))
            f.close()

            f = open(os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp/addresses", 'a')
            f.write(str(model_address)+'\n')
            f.close()
            model_address = sum(1 for line in open(os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp/addresses"))
        except:
            logger.info("Can not save this model")
    return model_address
            

def create_cv(smiles, split_type, n_cv, random_state):
    if split_type == "scaffold":
        count = n_cv
        n_cv = [([], []) for _ in range(count)]
        for i in range(count):
            train, test = scaffold_split(smiles, frac_train = 1 - ((len(smiles) / count) / (len(smiles)/100))/100, seed=random_state)
            n_cv[i][0].append(train)
            n_cv[i][1].append(test)
    
    if split_type == "cluster":
        count = n_cv        
        n_cv = [([], []) for _ in range(count)]
        for i in range(count):
            train, test = cluster_split(smiles, test_cluster_id=i, n_splits=count)
            n_cv[i][0].append(train)
            n_cv[i][1].append(test)
    return n_cv

    
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
    parser.add_argument('--targets', '-t', default=0, type=int, help='set number of target column'),
    parser.add_argument('--experiments_file', '-e', default='experiments.csv', help='where to write results of experiments')
    return parser

    
def script(args_list, random_state=False, p_rparams=False, verbose=0):
    time_start = datetime.now()

    # processing parameters
    args_list= list(map(str, args_list))
    if len(sys.argv) > 1:
        options = get_options().parse_args()
    else:
        options = get_options().parse_args(args_list)
    if options.targets is not list:
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

    # writing log to terminal (for stdout `stream=sys.stdout`)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    start_log(logger, options.gridsearch, options.n_bits, options.model_config, options.section, options.descriptors)
    # load data and configs
    root_address = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
    epochs, rparams, gparams = read_model_config(root_address+options.model_config, options.section)
    x_train, x_test, x_val, y_val, y_train, y_test, input_shape, output_shape, smiles = get_data(logger, root_address+options.data_config, options.n_bits, 
                                                                                                options.targets, random_state, options.split_type, options.split_s, 
                                                                                                verbose, options.descriptors, options.n_jobs)

    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2 or len(np.unique(y_val)) != 2 and "roc_auc" in options.metric:
        logger.error("Multiclass data: can not use ROC AUC metric")
        sys.exit(0)

    history = False
    model_loaded = False

    if options.load_model:
        model, model_loaded = load_model(options.load_model, logger)

    if options.gridsearch and not p_rparams and not model_loaded:
        logger.info("GRID SEARCH")  
        #scoring = {'accuracy': 'accuracy', 'MCC': make_scorer(matthews_corrcoef)}
        scoring = make_scoring(options.metric)
        ####
        loaded_cv = cv_splits_load(options.split_type, root_address+options.data_config)
        if loaded_cv is False:
            options.n_cv = create_cv(smiles, options.split_type, options.n_cv, random_state)
        else:
            options.n_cv = eval(loaded_cv)

        cv_splits_save(options.split_type, options.n_cv, root_address+options.data_config)
        f = open(options.output+'n_cv', 'w')
        f.write(str(options.n_cv))
        f.close()
        ####
        # check if number of iterations more then possible combinations
        keys = list(gparams.keys())
        n_iter = 1
        for k in keys:
            n_iter *= len(gparams[k])
        if options.n_iter > n_iter:
            options.n_iter = n_iter

        sklearn_params = {'param_distributions':gparams, 
                         'n_iter':options.n_iter, 
                         'n_jobs':options.n_jobs, 
                         'cv':options.n_cv, 
                         'verbose':verbose, 
                         'scoring':scoring, 
                         'random_state':random_state}

        randomized_params = {'param_distributions':gparams, 
                            'n_jobs':options.n_jobs, 
                            'cv':options.n_cv, 
                            'n_iter':options.n_iter, 
                            'verbose':verbose, 
                            'scoring':scoring, 
                            'random_state':random_state,
                            'return_train_score':True,
                            'pre_dispatch':n_cv}

        # sklearn models
        if options.select_model == "logreg":
            model = RandomizedSearchCV(LogisticRegression(**rparams), **sklearn_params)
        elif options.select_model == "knn":
            model = RandomizedSearchCV(KNeighborsClassifier(**rparams), **sklearn_params)
        elif options.select_model == "xgb" and xgb_flag:
            model = RandomizedSearchCV(xgb.XGBClassifier(**rparams), **sklearn_params)
        elif options.select_model == "svc":
            model = RandomizedSearchCV(SVC(**rparams), **sklearn_params)
        elif options.select_model == "rf":
            model = RandomizedSearchCV(RandomForestClassifier(**rparams), **sklearn_params)
        elif options.select_model == "if":
            model = RandomizedSearchCV(IsolationForest(**rparams), **sklearn_params)                                       
        # keras models        
        elif options.select_model == "regression":
            search_model = KerasClassifier(build_fn=Logreg, input_shape=input_shape, output_shape=output_shape)
            model = RandomizedSearchCV(estimator=search_model, **randomized_params)
        elif options.select_model == "residual":
            search_model = KerasClassifier(build_fn=Residual, input_shape=input_shape, output_shape=output_shape)
            model = RandomizedSearchCV(estimator=search_model, **randomized_params)                                    
        elif options.select_model == "lstm":
            search_model = KerasClassifier(build_fn=LSTM, input_shape=input_shape, output_shape=output_shape, input_length=x_train.shape[1])
            model = RandomizedSearchCV(estimator=search_model, **randomized_params)                                    
        elif options.select_model == "rnn":
            search_model = KerasClassifier(build_fn=RNN, input_shape=input_shape, output_shape=output_shape, input_length=x_train.shape[1])
            model = RandomizedSearchCV(estimator=search_model, **randomized_params)        
        elif options.select_model == "gru":
            search_model = KerasClassifier(build_fn=GRU, input_shape=input_shape, output_shape=output_shape, input_length=x_train.shape[1])
            model = RandomizedSearchCV(estimator=search_model, **randomized_params)
        else:
            logger.info("Model name is not found.")
            sys.exit(0)
        
        time.sleep(5)
        logger.info("GRIDSEARCH FIT")
        model.fit(x_train, np.ravel(y_train))
        rparams = model.best_params_
        grid = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending=False)
        grid.to_csv(options.output + "gridsearch.csv")

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

    if p_rparams:
        rparams = p_rparams
    elif options.select_model == "logreg":
        model = LogisticRegression(**rparams)
    elif options.select_model == "knn":
        model = KNeighborsClassifier(**rparams)
    elif options.select_model == "xgb":
        model = xgb.XGBClassifier(**rparams)
    elif options.select_model == "svc":
        model = SVC(**rparams)
    elif options.select_model == "rf":
        model = RandomForestClassifier(**rparams)
    elif options.select_model == "if":
        model = IsolationForest(**rparams)

    elif options.select_model == "residual":
        model = Residual(input_shape, output_shape, **rparams)
    elif options.select_model == "regression":
        model = Logreg(input_shape, output_shape, **rparams)
    elif options.select_model == "rnn":
        model = RNN(input_shape, output_shape, **rparams)
    elif options.select_model == "gru":
        model = GRU(input_shape, output_shape, **rparams)
    elif options.select_model == "lstm":
        model = LSTM(input_shape, output_shape, **rparams)
    else:
        logger.info("Model name is not found or xgboost import error.")
        sys.exit(0)

    logger.info("MODEL FIT")
    try:
        print("NORMAL FITTING")
        callbacks = create_callbacks(options.output, options.patience, options.section)
        history = model.fit(x_train, np.ravel(y_train), batch_size=rparams.get("batch_size"), epochs=epochs, shuffle=False, verbose=verbose, callbacks=callbacks, validation_data=(x_val, y_val))
    except:
        model.fit(x_train, np.ravel(y_train))
    
    logger.info("EVALUATE")
    accuracy_test, accuracy_train, rec, auc, auc_val, f1, path = evaluate(logger, options, random_state, options.output, model, x_train, 
                                                                    x_test, x_val, y_val, y_train, y_test, time_start, rparams, history, 
                                                                    options.section, options.n_jobs, descriptors)
    model_address = save_model(model, path, logger)

    logger.info("Done")
    return accuracy_test, accuracy_train, rec, auc, auc_val, f1, rparams, model_address


if __name__ == "__main__":
    args_list = ['logreg', '/data/data_configs/bace.ini']
    script(args_list)
