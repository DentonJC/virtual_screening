#!/usr/bin/env python

import os
import logging
import argh
import numpy as np
from argh.decorators import arg
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from src.main import create_callbacks, read_config, evaluate, start_log
from src.gridsearch import grid_search
from src.data_loader import get_data
from sklearn.metrics import matthews_corrcoef, make_scorer
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Merge,TimeDistributed, merge, GRU, SimpleRNN
mcc = make_scorer(matthews_corrcoef)
time_start = datetime.now()
n_physical = 196


def lstm(input_dim = 28, MAX_SEQ_LENGTH=None, N_CLASSES=2):
    print("FORWARD")
    encoder_a = Sequential()
    encoder_a.add(LSTM(8, input_dim=input_dim,return_sequences=True))
    print("BACKWARD")
    encoder_b = Sequential()
    encoder_b.add(LSTM(8, input_dim=input_dim,go_backwards=True,return_sequences=True))

    print("MODEL")
    model = Sequential()
    model.add(Merge([encoder_a, encoder_b], mode='concat'))
    model.add(TimeDistributed(Dense(N_CLASSES, activation='softmax')))

    print("COMPILE")
    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    return model


    
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
    
    
    if gridsearch:
        try:
            model = RandomizedSearchCV(lstm(x_train.shape[1]), gparams, n_iter=n_iter, n_jobs=n_jobs, cv=n_folds, verbose=10, scoring=[mcc, 'f1_weighted', 'precision_weighted', 'r2', 'recall_weighted'])
            print("FIT")
            logging.info("FIT")
            model.fit(x_train, np.ravel(y_train))
        except:
            model = RandomizedSearchCV(lstm(x_train.shape[1]), gparams, n_iter=n_iter, n_jobs=n_jobs, cv=n_folds, verbose=10)
            print("FIT")
            logging.info("FIT")
            model.fit(x_train, np.ravel(y_train))
    else:
        model = lstm(x_train.shape[1])
        print("FIT")
        logging.info("FIT")
        model.fit(x_train, np.ravel(y_train))

    if gridsearch:
        rparams = model.cv_results_
        print(rparams)

    print("EVALUATE")
    logging.info("EVALUATE")
    train_acc, test_acc = evaluate(output, model, x_train, x_test, x_val, y_train, y_test, y_val, time_start, rparams, None)
    with open(os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", "") + "/tmp/last_result", 'w') as tmp_file:
        tmp_file.write(str(train_acc) + '\n' + str(test_acc))
    

parser = argh.ArghParser()
argh.set_default_command(parser, main)

if __name__ == "__main__":
    parser.dispatch()
