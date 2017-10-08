#!/usr/bin/env python

import os
import sys
import logging
import numpy as np
import argh
from argh.decorators import arg
from datetime import datetime
from sklearn.utils import class_weight as cw
from src.main import create_callbacks, read_config, evaluate, start_log
from src.gridsearch import grid_search
from src.data import get_data
from src.models.models import build_residual_model
from src.report import auc
time_start = datetime.now()
n_physical = 196


def main(
    data:'path to dataset',
    section:'name of section in config file',
    # targets:'set target'=0,
    features:'take features: all, fingerptint or physical'='all',
    output:'path to output directory'=os.path.dirname(os.path.realpath(__file__)).replace("/src/scripts", "") + "/tmp/" + str(time_start) + '/',
    configs:'path to config file'=os.path.dirname(os.path.realpath(__file__)).replace("/scripts", "") + "/configs/configs.ini",
    fingerprint:'maccs (167) or morgan (n)'='morgan',
    n_bits:'number of bits in Morgan fingerprint'=256,
    n_jobs:'number of jobs'=1,
    patience:'patience of fit'=100,
    gridsearch:'use gridsearch'=False,
    dummy:'use only first 1000 rows of dataset'=False,
    ):

    callbacks_list = create_callbacks(output, patience, data)
    logging.basicConfig(filename=output+'main.log', level=logging.INFO)
    start_log(dummy, gridsearch, fingerprint, n_bits, configs, data, section)
    n_folds, epochs, rparams, gparams, n_iter, class_weight, targets = read_config(configs, section)
    x_train, x_test, x_val, y_train, y_test, y_val, input_shape, output_shape, smiles = get_data(data, dummy, fingerprint, n_bits, targets, features)
    
    if gridsearch:
        rparams = grid_search(gparams, build_residual_model, x_train, y_train, input_shape, output_shape, output, n_folds, n_iter, n_jobs)

    model = build_residual_model(input_shape, output_shape, activation_0=rparams.get("activation_0", 'softmax'), activation_1=rparams.get("activation_0", 'softmax'), activation_2=rparams.get("activation_0", 'softmax'),
                                 loss=rparams.get("loss", 'binary_crossentropy'), metrics=rparams.get("metrics", ['accuracy']),
                                 optimizer=rparams.get("optimizer", 'Adam'), learning_rate=rparams.get("learning_rate", 0.001),
                                 momentum=rparams.get("momentum", 0.1), init_mode=rparams.get("init_mode", 'uniform'), dropout=rparams.get("dropout", 0), layers=rparams.get("layers", 0))

    print("FIT")
    logging.info("FIT")
    if not class_weight:
        y = [item for sublist in y_train for item in sublist]
        class_weight = cw.compute_class_weight("balanced", np.unique(y), y)
    history = model.fit(x_train, y_train, batch_size=rparams.get("batch_size"), epochs=epochs, validation_data=(x_val, y_val), shuffle=True, verbose=1, callbacks=callbacks_list, class_weight=class_weight)
    print("EVALUATE")
    logging.info("EVALUATE")
    evaluate(output, model, x_train, x_test, x_val, y_train, y_test, y_val, time_start, rparams, history)
    auc(model, x_train, x_test, x_val, y_train, y_test, y_val, output)


parser = argh.ArghParser()
argh.set_default_command(parser, main)

if __name__ == "__main__":
    parser.dispatch()
