#!/usr/bin/env python

import os
import logging
import numpy as np
import argh
from argh.decorators import arg
from datetime import datetime
from sklearn.utils import class_weight as cw
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from src.main import create_callbacks, read_config, evaluate, start_log
from src.data_loader import get_data
from src.experiment import write_experiment
from sklearn.metrics import matthews_corrcoef, make_scorer
mcc = make_scorer(matthews_corrcoef)
time_start = datetime.now()
n_physical = 196


def main(
    data:'path to dataset',
    section:'name of section in config file',
    features:'take features: all, fingerptint or physical'='all',
    output:'path to output directory'=os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", "") + "/tmp/" + str(time_start) + '/',
    configs:'path to config file'=os.path.dirname(os.path.realpath(__file__)).replace("/scripts", "") + "/configs/configs.ini",
    fingerprint:'maccs (167) or morgan (n)'='morgan',
    n_bits:'number of bits in Morgan fingerprint'=256,
    n_jobs:'number of jobs'=1,
    patience:'patience of fit'=100,
    gridsearch:'use gridsearch'=False,
    dummy:'use only first 1000 rows of dataset'=False,
    targets: 'set number of target column'=0,
    experiments_file: 'where to write results of experiments'='experiments.csv'
    ):
    if targets is not list:
        targets = [targets]

    callbacks_list = create_callbacks(output, patience, data)
    logging.basicConfig(filename=output+'main.log', level=logging.INFO)
    start_log(dummy, gridsearch, fingerprint, n_bits, configs, data, section)
    n_folds, epochs, rparams, gparams, n_iter, class_weight = read_config(configs, section)
    x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles = get_data(data, dummy, fingerprint, n_bits, targets, features)
    
    if not class_weight:
        y = [item for sublist in y_train for item in sublist]
        class_w = cw.compute_class_weight("balanced", np.unique(y), y)
        class_weight = {0: class_w[0], 1: class_w[1]}

    if gridsearch:
        try:
            model = RandomizedSearchCV(SVC(**rparams), gparams, n_iter=n_iter, n_jobs=n_jobs, cv=n_folds, verbose=10, scoring=[mcc, 'f1_weighted', 'precision_weighted', 'r2', 'recall_weighted'])
            print("FIT")
            logging.info("FIT")
            history = model.fit(x_train, np.ravel(y_train))
        except:
            model = RandomizedSearchCV(SVC(**rparams), gparams, n_iter=n_iter, n_jobs=n_jobs, cv=n_folds, verbose=10)
            print("FIT")
            logging.info("FIT")
            history = model.fit(x_train, np.ravel(y_train))
    else:
        model = SVC(**rparams, class_weight="balanced")
        print("FIT")
        logging.info("FIT")
        history = model.fit(x_train, np.ravel(y_train))

    if gridsearch:
        rparams = model.cv_results_
        print(rparams)

    print("EVALUATE")
    logging.info("EVALUATE")
    train_acc, test_acc = evaluate(output, model, x_train, x_test, x_val, y_train, y_test, y_val, time_start, rparams, history)
    write_experiment(train_acc, test_acc, targets, experiments_file)


parser = argh.ArghParser()
argh.set_default_command(parser, main)

if __name__ == "__main__":
    parser.dispatch()
