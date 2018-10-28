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
from moloi.model_processing import save_model
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


def evaluate(logger, options, random_state, model, data, time_start, rparams, history, score, results, plots):
    path = options.output
    y_pred_test = model.predict(data["x_test"])
    y_pred_train = model.predict(data["x_train"])
    save_labels(data["y_test"], path + "y_test.csv")
    save_labels(y_pred_test, path + "y_pred_test.csv")
    y_pred_val = model.predict(data["x_val"])
    save_labels(data["y_val"], path + "y_val.csv")
    save_labels(y_pred_val, path + "y_pred_val.csv")

    y_pred_test = np.ravel(y_pred_test)
    y_pred_train = np.ravel(y_pred_train)
    y_pred_val = np.ravel(y_pred_val)

    try:
        y_pred_test = [int(round(value)) for value in y_pred_test]
        y_pred_train = [int(round(value)) for value in y_pred_train]
        y_pred_val = [int(round(value)) for value in y_pred_val]
    except ValueError:
        logger.error("Model is not trained")
        print(y_pred_test)
        sys.exit(0)

    try:
        accuracy_test = accuracy_score(list(np.ravel(data["y_test"])), y_pred_test)*100
        accuracy_train = accuracy_score(list(np.ravel(data["y_train"])), y_pred_train)*100
        accuracy_val = accuracy_score(list(np.ravel(data["y_val"])), y_pred_val)*100
        logger.info("Accuracy test: %.2f%%" % (accuracy_test))

        f1_test = f1_score(data["y_test"], y_pred_test, average=None)
        f1_train = f1_score(data["y_train"], y_pred_train, average=None)
        f1_val = f1_score(data["y_val"], y_pred_val, average=None)

        rec_test = recall_score(data["y_test"], y_pred_test, average=None)
        rec_train = recall_score(data["y_train"], y_pred_train, average=None)
        rec_val = recall_score(data["y_val"], y_pred_val, average=None)
    except ValueError:
        accuracy_test = False
        accuracy_val = False
        accuracy_train = False
        f1_test = False
        f1_train = False
        f1_val = False
        rec_test = False
        rec_train = False
        rec_val = False

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
        auc_train = False
        auc_test = False
        auc_val = False
        train_proba = False
        test_proba = False
        val_proba = False

    try:
        rmse_train = sqrt(abs(mean_squared_error(data["y_train"], y_pred_train)))
        rmse_test = sqrt(abs(mean_squared_error(data["y_test"], y_pred_test)))
        rmse_val = sqrt(abs(mean_squared_error(data["y_val"], y_pred_val)))

        mae_train = mean_absolute_error(data["y_train"], y_pred_train)
        mae_test = mean_absolute_error(data["y_test"], y_pred_test)
        mae_val = mean_absolute_error(data["y_val"], y_pred_val)

        r2_train = r2_score(data["y_train"], y_pred_train)
        r2_test = r2_score(data["y_test"], y_pred_test)
        r2_val = r2_score(data["y_val"], y_pred_val)
    except ValueError:
        rmse_train = False
        rmse_test = False
        rmse_val = False
        mae_train = False
        mae_test = False
        mae_val = False
        r2_train = False
        r2_test = False
        r2_val = False

    results = {
        'accuracy_test': accuracy_test,
        'accuracy_train': accuracy_train,
        'accuracy_val': accuracy_val,
        'rec_train': rec_train,
        'rec_test': rec_test,
        'rec_val': rec_val,
        'auc_test': auc_test,
        'auc_train': auc_train,
        'auc_val': auc_val,
        'f1_train': f1_train,
        'f1_test': f1_test,
        'f1_val': f1_val,
        'rmse_test': rmse_test,
        'rmse_train': rmse_train,
        'rmse_val': rmse_val,
        'mae_test': mae_test,
        'mae_train': mae_train,
        'mae_val': mae_val,
        'r2_test': r2_test,
        'r2_train': r2_train,
        'r2_val': r2_val,
        'rparams': rparams
        }

    try:
        results["balanced_accuracy_test"] = (results["rec_test"][0] + results["rec_test"][1]) / 2
        results["balanced_accuracy_train"] = (results["rec_train"][0] + results["rec_train"][1]) / 2
        results["balanced_accuracy_val"] = (results["rec_val"][0] + results["rec_val"][1]) / 2
    except TypeError:
        results["balanced_accuracy_test"] = False
        results["balanced_accuracy_train"] = False
        results["balanced_accuracy_val"] = False
    # find how long the program was running
    tstop = datetime.now()
    timer = tstop - time_start
    logger.info(timer)
    # create report, prediction and save script and all current models
    create_report(logger, path, train_proba, test_proba, val_proba, timer, rparams,
                  time_start, history, random_state, options, data, y_pred_train,
                  y_pred_test, y_pred_val, score, model, results, plots)

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
    
    # transfer learning in row
    if not options.load_model:
        model_address = save_model(model, path, logger, results['rparams'])
    else:
        model_address = options.load_model
        
    results['model_address'] = model_address
    
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
