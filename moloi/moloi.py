#!/usr/bin/env python

# TODO: fix docstrings

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler  # , Normalizer
from moloi.config_processing import read_model_config
from moloi.evaluation import evaluate, make_scoring
# from moloi.splits.cv import create_cv_splits
from moloi.data_processing import get_data, clean_data
from moloi.model_processing import load_model, save_model
from moloi.arguments import get_options
from moloi.create_model import create_model, create_gridsearch_model
from moloi.models.keras_models import create_callbacks


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
    # options.descriptors = eval(options.descriptors)  # input is string, need array
    options.descriptors = options.descriptors.split()
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

    if not exp_settings["logger_flag"]:
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

    epochs, rparams, gparams = read_model_config(options.model_config, options.section)

    data = get_data(logger, options, exp_settings["random_state"], exp_settings["verbose"])
    # x_train, x_test, x_val, y_val, y_train, y_test, smiles
    data["x_train"] = clean_data(data["x_train"])
    data["x_test"] = clean_data(data["x_test"])
    data["x_val"] = clean_data(data["x_val"])

    # Scale
    if options.select_model == "svc":
        transformer_X = MinMaxScaler(feature_range=(-1, 1)).fit(data["x_train"])
    else:
        transformer_X = MinMaxScaler().fit(data["x_train"])
    data["x_train"] = transformer_X.transform(data["x_train"])
    data["x_test"] = transformer_X.transform(data["x_test"])
    data["x_val"] = transformer_X.transform(data["x_val"])

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

    if options.load_model:
        try:
            _, model_loaded = load_model(options.load_model, logger)
        except:
            logger.info("Can not load model")

    if options.gridsearch and not exp_settings["rparams"] and not model_loaded:
        logger.info("GRID SEARCH")
        logger.info("x_train shape: %s", str(np.array(data["x_train"]).shape))
        logger.info("x_test shape: %s", str(np.array(data["x_test"]).shape))
        logger.info("x_val shape: %s", str(np.array(data["x_val"]).shape))
        logger.info("y_train shape: %s", str(np.array(data["y_train"]).shape))
        logger.info("y_test shape: %s", str(np.array(data["y_test"]).shape))
        logger.info("y_val shape: %s", str(np.array(data["y_val"]).shape))
        # scoring = {'accuracy': 'accuracy', 'MCC': make_scorer(matthews_corrcoef)}
        scoring = make_scoring(options.metric)
        # options.n_cv = create_cv_splits()  # disabled
        # check if number of iterations more then possible combinations
        keys = list(gparams.keys())
        n_iter = 1
        for k in keys:
            n_iter *= len(gparams[k])
        if options.n_iter > n_iter:
            options.n_iter = n_iter
        model = create_gridsearch_model(logger, rparams, gparams, options, exp_settings, scoring, options.n_cv,
                                        data["x_train"].shape[1], data["y_train"].shape[1])
        logger.info("GRIDSEARCH FIT")
        if data["y_train"].shape[1] == 1:
            model.fit(data["x_train"], np.ravel(data["y_train"]))
        else:
            model.fit(data["x_train"], data["y_train"])
        rparams = model.best_params_
        grid = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending=False)
        grid.to_csv(options.output + "gridsearch.csv")

    if exp_settings["rparams"]:
        rparams = exp_settings["rparams"]

    params_for_sklearn = dict(rparams)
    try:
        del params_for_sklearn["epochs", "class_weight", "batch_size"]
    except KeyError:
        pass

    model = create_model(logger, params_for_sklearn, options, data["x_train"].shape[1], data["y_train"].shape[1])

    if options.load_model:
        model, model_loaded = load_model(options.load_model, logger)

    if (options.load_model and exp_settings["refit"]) or not options.load_model:
        logger.info("MODEL FIT")
        # TODO: replace try by if keras model
        try:
            monitor = 'binary_crossentropy'
            mode = 'auto'
            callbacks = create_callbacks(options.output, options.patience, options.section,
                                         monitor="val_" + monitor, mode=mode)
            history = model.fit(data["x_train"], np.ravel(data["y_train"]),
                                batch_size=rparams.get("batch_size"),
                                epochs=epochs, shuffle=False,
                                verbose=exp_settings["verbose"],
                                callbacks=callbacks,
                                validation_data=(data["x_val"], data["y_val"]))
        except:
            model.fit(data["x_train"], np.ravel(data["y_train"]))

    # transfer learning in row
    if not options.load_model:
        model_address = save_model(model, options.output, logger, results['rparams'])
    else:
        model_address = options.load_model

    logger.info("EVALUATE")
    results, path = evaluate(logger, options, exp_settings["random_state"], model, data,
                             time_start, rparams, history, grid, results)

    logger.info("Done")

    results['model_address'] = model_address
    return results


if __name__ == "__main__":
    args_list = ['lr', '/data/data_configs/bace.ini']
    experiment(args_list)
