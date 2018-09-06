#!/usr/bin/env python

# TODO: replace try by if
# TODO: fix docstrings

import os
import sys
import glob
from shutil import copyfile, copytree
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, matthews_corrcoef, make_scorer
from moloi.report import create_report


def get_latest_file(path):
    """
    Return the path to the last (and best) checkpoint.
    """
    list_of_files = glob.iglob(path + "results/*")
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    _, filename = os.path.split(latest_file)

    return path+"results/"+filename


def save_labels(arr, filename):
    pd_array = pd.DataFrame(arr)
    pd_array.index.names = ["Id"]
    pd_array.columns = ["Prediction"]
    pd_array.to_csv(filename)


def make_scoring(metric):
    scoring = make_scorer(accuracy_score)
    if metric in 'accuracy':
        scoring = make_scorer(accuracy_score)
    if metric in 'roc_auc':
        scoring = make_scorer(roc_auc_score)
    if metric in 'f1':
        scoring = make_scorer(f1_score)
    if metric in 'matthews':
        scoring = make_scorer(matthews_corrcoef)
    return scoring


def evaluate(logger, options, random_state, model, data, time_start, rparams, history, score, results):
    path = options.output
    y_pred_test = model.predict(data["x_test"])
    y_pred_train = model.predict(data["x_train"])
    save_labels(y_pred_train, path + "y_pred_test.csv")
    y_pred_val = model.predict(data["x_val"])
    save_labels(y_pred_val, path + "y_pred_val.csv")

    y_pred_test = np.ravel(y_pred_test)
    y_pred_train = np.ravel(y_pred_train)
    y_pred_val = np.ravel(y_pred_val)

    try:
        y_pred_test = [int(round(value)) for value in y_pred_test]
        y_pred_train = [int(round(value)) for value in y_pred_train]
    except ValueError:
        logger.error("Model is not trained")
        print(y_pred_test)
        sys.exit(0)

    accuracy_test = accuracy_score(list(np.ravel(data["y_test"])), y_pred_test)*100
    accuracy_train = accuracy_score(list(np.ravel(data["y_train"])), y_pred_train)*100
    logger.info("Accuracy test: %.2f%%" % (accuracy_test))

    rec = recall_score(data["y_test"], y_pred_test, average=None)
    try:
        train_proba = model.predict_proba(data["x_train"])
        test_proba = model.predict_proba(data["x_test"])
        val_proba = model.predict_proba(data["x_val"])
        try:
            train_proba = train_proba[:, 1]
            val_proba = val_proba[:, 1]
            test_proba = test_proba[:, 1]
        except:
            pass
        auc_train = roc_auc_score(data["y_train"], train_proba)
        auc_test = roc_auc_score(data["y_test"], test_proba)
        auc_val = roc_auc_score(data["y_val"], val_proba)
    except:
        e = sys.exc_info()[0]
        print("Error: %s" % e)
        auc_train = False
        auc_test = False
        auc_val = False
        train_proba = False
        test_proba = False
        val_proba = False

    f1 = f1_score(data["y_test"], y_pred_test, average=None)
    results = {
        'accuracy_test': accuracy_test,
        'accuracy_train': accuracy_train,
        'rec': rec,
        'auc_test': auc_test,
        'auc_train': auc_train,
        'auc_val': auc_val,
        'f1': f1,
        'rparams': rparams
        }
    # find how long the program was running
    tstop = datetime.now()
    timer = tstop - time_start
    logger.info(timer)
    # create report, prediction and save script and all current models
    create_report(logger, path, train_proba, test_proba, val_proba, timer, rparams,
                  time_start, history, random_state, options, data, y_pred_train,
                  y_pred_test, y_pred_val, score, model, results)

    copyfile(sys.argv[0], path + os.path.basename(sys.argv[0]))
    try:
        copytree('moloi/models', path + 'models')
    except:  # FileNotFoundError not comp python2
        pass

    path_old = path[:-1]

    try:
        path = (path[:-8] + '_' + options.section + '_' + str(options.descriptors) +
                '_' + str(round(accuracy_test, 3)) + '/').replace(" ", "_")
        os.rename(path_old, path)
    except TypeError:
        pass

    logger.info("Results path: %s", path)
    return results, path
