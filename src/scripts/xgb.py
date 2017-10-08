#!/usr/bin/env python

import os
import logging
import xgboost as xgb
import argh
from argh.decorators import arg
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from src.main import create_callbacks, read_config, evaluate, start_log
from src.data import get_data
time_start = datetime.now()


def main(
    data:'path to dataset',
    section:'name of section in config file',
    targets:'set targets'=0, 
    features:'set features'=256, 
    output:'path to output directory'=os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", "") + "/tmp/" + str(time_start) + '/',
    configs:'path to config file'=os.path.dirname(os.path.realpath(__file__)).replace("/scripts", "") + "/configs/configs.ini", 
    fingerprint:'maccs (167) or morgan (n)'='morgan', 
    n_bits:'number of bits in Morgan fingerprint'=256, 
    n_jobs:'number of jobs'=1, 
    patience:'patience of fit'=100, 
    gridsearch:'use gridsearch'=False, 
    dummy:'use only first 1000 rows of dataset'=False,
    ):
    
    features = range(0, features)

    callbacks_list = create_callbacks(output, patience, data)
    logging.basicConfig(filename=output+'main.log', level=logging.INFO)
    start_log(dummy, gridsearch, fingerprint, n_bits, configs, data, section)
    n_folds, epochs, rparams, gparams, n_iter, class_weight = read_config(configs, section)
    x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles = get_data(data, dummy, fingerprint, n_bits, targets, features)
    
    if gridsearch:
        model = GridSearchCV(xgb.XGBClassifier(**rparams), gparams, scoring='accuracy', cv=n_folds, n_jobs=n_jobs, verbose=10)
    else:
        model = xgb.XGBClassifier(**rparams)

    print("FIT")
    logging.info("FIT")
    model.fit(x_train, y_train.reshape(y_train.shape[0],))

    if gridsearch:
        rparams = model.grid_scores_
        print(rparams)

    print("EVALUATE")
    logging.info("EVALUATE")
    evaluate(output, model, x_train, x_test, x_val, y_train, y_test, y_val, time_start, rparams, None)


parser = argh.ArghParser()
argh.set_default_command(parser, main)

if __name__ == "__main__":
    parser.dispatch()
