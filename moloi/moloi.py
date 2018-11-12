#!/usr/bin/env python

# TODO: fix docstrings

import os
import sys
import csv
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from moloi.evaluation import evaluate, make_scoring
from moloi.splits.cv import create_cv_splits
from moloi.data_processing import get_data, clean_data
from moloi.model_processing import load_model
from moloi.arguments import get_options
from moloi.create_pipeline import create_pipeline
from moloi.models.keras_models import create_callbacks
from moloi.generate_plots import generate_decomposition


def experiment(args_list, exp_settings, results):
    time_start = datetime.now()

    # processing parameters
    args_list = list(map(str, args_list))
    if len(sys.argv) > 1:
        options = get_options().parse_args()
    else:
        options = get_options().parse_args(args_list)

    for i in range(len(options.targets)):  # options.targets is array of strings
        options.targets[i] = int(options.targets[i])

    options.descriptors = options.descriptors.split(',')
    for i, desc in enumerate(options.descriptors):
        options.descriptors[i] = desc.replace('\'','').replace('[','').replace(']','').replace(' ','')
        
    options.select_model = options.select_model.split(',')
    for i, model in enumerate(options.select_model):
        options.select_model[i] = model.replace('\'','').replace('[','').replace(']','').replace(' ','')
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

    if len(list(logger.handlers)) == 1:
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
    options.data_config = root_address + options.data_config
    options.model_config = root_address + options.model_config
    
    # scoring = {'accuracy': 'accuracy', 'MCC': make_scorer(matthews_corrcoef)}
    scoring = make_scoring(options.metric)     

    data = get_data(logger, options, exp_settings["random_state"], exp_settings["verbose"])
    data["x_train"] = clean_data(data["x_train"])
    data["x_test"] = clean_data(data["x_test"])
    data["x_val"] = clean_data(data["x_val"])

    if len(np.unique(data["y_train"])) == 1:
        logger.error("Multiclass data: only one class in y_train")
        sys.exit(0)
    if len(np.unique(data["y_test"])) == 1:
        logger.error("Multiclass data: only one class in y_test")
        sys.exit(0)
    if len(np.unique(data["y_val"])) == 1:
        logger.error("Multiclass data: only one class in y_val")
        sys.exit(0)
    if (len(np.unique(data["y_train"])) > 2 or len(np.unique(data["y_test"])) > 2 or
       len(np.unique(data["y_val"])) > 2) and "roc_auc" in options.metric:
        logger.error("Multiclass data: can not use ROC AUC metric")
        sys.exit(0)
    if data["y_train"].shape[1] > 1 and "roc_auc" in options.metric:
        logger.error("Multilabel data: can not use ROC AUC metric")
        sys.exit(0)

    history = False
    model_loaded = False
    grid = False

    rparams = False
    if options.load_model:
        model, rparams, model_loaded = load_model(options.load_model, logger)

    if options.gridsearch and not model_loaded:
        logger.info("GRID SEARCH")
        model, epochs, data, rparams = create_pipeline(logger, options, exp_settings, scoring, data, gridsearch=True)
        
        try:
            monitor = 'binary_crossentropy'
            monitor = 'acc'
            mode = 'auto'
            callbacks = create_callbacks(options.output, options.patience, options.section,
                                         monitor="val_" + monitor, mode=mode, callbacks=exp_settings["callbacks"])

            history = model.fit(data["x_train"], data["y_train"],
                                batch_size=rparams.get("batch_size"),
                                epochs=epochs, shuffle=False,
                                verbose=exp_settings["verbose"],
                                callbacks=callbacks,
                                validation_data=(data["x_val"], data["y_val"]))
        except:
            history = model.fit(data["x_train"], np.ravel(data["y_train"]))

        rparams = model.best_params_
        grid = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending=False)
        grid.to_csv(options.output + "gridsearch.csv", index=False)
        try:
            history = history.best_estimator_.model.history.history
            df = pd.DataFrame(history)
            df.to_csv(options.output + "history.csv", index=False, sep=';')
        except:
            pass
        model = model.best_estimator_
    
    elif not options.gridsearch:
        model, epochs, data, rparams = create_pipeline(logger, options, exp_settings, scoring, data, gridsearch=False)

    if exp_settings["refit"] or not options.gridsearch:
        logger.info("MODEL FIT")
        try:
            monitor = 'binary_crossentropy'
            monitor = 'acc'
            mode = 'auto'
            callbacks = create_callbacks(options.output, options.patience, options.section,
                                         monitor="val_" + monitor, mode=mode, callbacks=exp_settings["callbacks"])

            history = model.fit(data["x_train"], data["y_train"],
                                batch_size=rparams.get("batch_size"),
                                epochs=epochs, shuffle=False,
                                verbose=exp_settings["verbose"],
                                callbacks=callbacks,
                                validation_data=(data["x_val"], data["y_val"]))
        except:
            model.fit(data["x_train"], np.ravel(data["y_train"]))
        
    logger.info("EVALUATE")
    generate_decomposition([exp_settings["experiments_file"]], [str(options.descriptors)], [options.split_type], options, exp_settings["random_state"], exp_settings["verbose"])
    results, path = evaluate(logger, options, exp_settings["random_state"], model, data,
                             time_start, rparams, history, grid, results, exp_settings["plots"])
    logger.info("Done")
    return results


if __name__ == "__main__":
    args_list = ['lr', '/data/data_configs/bace.ini']
    experiment(args_list)
