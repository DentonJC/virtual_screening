#!/usr/bin/env python

# TODO: replace try by if
# TODO: fix docstrings

import os
import shutil
import sys
import glob
from shutil import copyfile, copytree
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, r2_score
from sklearn.metrics import matthews_corrcoef, make_scorer, mean_squared_error, mean_absolute_error
from moloi.report import create_report
from math import sqrt


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
    if metric in 'accuracy':
        scoring = make_scorer(accuracy_score)
    elif metric in 'roc_auc':
        scoring = make_scorer(roc_auc_score)
    elif metric in 'f1':
        scoring = make_scorer(f1_score)
    elif metric in 'matthews':
        scoring = make_scorer(matthews_corrcoef)
    elif metric in 'mae':
        scoring = make_scorer(mean_absolute_error, greater_is_better=False)
    elif metric in 'r2':
        scoring = make_scorer(r2_score)
    else:
        scoring = make_scorer(accuracy_score)
    return scoring


def evaluate(logger, options, random_state, model, data, time_start, rparams, history, score, results):
    path = options.output
    y_pred_test = model.predict(data["x_test"])
    y_pred_train = model.predict(data["x_train"])
    save_labels(y_pred_test, path + "y_pred_test.csv")
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
    rmse_train = sqrt(mean_squared_error(data["y_train"], y_pred_train))
    rmse_test = sqrt(mean_squared_error(data["y_test"], y_pred_test))

    mae_train = mean_absolute_error(data["y_train"], y_pred_train)
    mae_test = mean_absolute_error(data["y_test"], y_pred_test)

    results = {
        'accuracy_test': accuracy_test,
        'accuracy_train': accuracy_train,
        'rec': rec,
        'auc_test': auc_test,
        'auc_train': auc_train,
        'auc_val': auc_val,
        'f1': f1,
        'rmse_test': rmse_test,
        'rmse_train': rmse_train,
        'mae_test': mae_test,
        'mae_train': mae_train,
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

    try:
        root = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp/"
        folders = list(os.walk(root))
        folders = folders[0][1]
        for folder in folders:
            	shutil.make_archive(os.path.join(root, folder), 'zip', os.path.join(root, folder))
            	shutil.rmtree(os.path.join(root, folder))
    except:
         pass
    logger.info("Results path: %s", path)
    return results, path
