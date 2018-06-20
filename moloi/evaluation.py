#!/usr/bin/env python

import os
import sys
import csv
import glob
import math
import pickle
import getopt
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot
from datetime import datetime
from shutil import copyfile, copytree
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef, make_scorer
from moloi.report import create_report, plot_auc, plot_TSNE
from moloi.plots import plot_fi
from moloi.descriptors.rdkit import rdkit_fetures_names
from moloi.descriptors.mordred import mordred_fetures_names
from moloi.data_processing import m_mean


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


def evaluate(logger, options, random_state, path, model, x_train, x_test, x_val, y_val, y_train, y_test, time_start, rparams, history, section, n_jobs, descriptors, score):
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    save_labels(y_pred_train, path + "y_pred_test.csv")
    y_pred_val = model.predict(x_val)
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
        
    accuracy_test = accuracy_score(list(np.ravel(y_test)), y_pred_test)*100
    accuracy_train = accuracy_score(list(np.ravel(y_train)), y_pred_train)*100
    logger.info("Accuracy test: %.2f%%" % (accuracy_test))
        
    rec = recall_score(y_test, y_pred_test, average=None)
    try:
        train_proba = model.predict_proba(x_train)
        test_proba = model.predict_proba(x_test)
        val_proba = model.predict_proba(x_val)
        try:
            train_proba = train_proba[:,1]
            val_proba = val_proba[:,1]
            test_proba = test_proba[:,1]
        except:
            pass
        auc_train = roc_auc_score(y_train, train_proba)
        auc_test = roc_auc_score(y_test, test_proba)
        auc_val = roc_auc_score(y_val, val_proba)
    except:
        e = sys.exc_info()[0]
        print("Error: %s" % e)
        auc_train = False
        auc_test = False
        auc_val = False
        train_proba = False
        test_proba = False
        val_proba = False

    f1 = f1_score(y_test, y_pred_test, average=None)
    """
    # balanced acc
    num_positive = float(np.count_nonzero(y_test))
    num_negative = float(len(y_test) - num_positive)
    pos_weight = num_negative / num_positive
    weights = np.ones_like(y_test)
    weights[y_train != 0] = pos_weight
    b_acc = accuracy_score(y_test, y_pred_test, sample_weight=weights)
    """
    
    # find how long the program was running
    tstop = datetime.now()
    timer = tstop - time_start
    logger.info(timer)
    # create report, prediction and save script and all current models
    create_report(logger, path, accuracy_test, accuracy_train, rec, auc_train, auc_test, auc_val, train_proba, test_proba, val_proba, f1, timer, rparams, time_start, history, random_state, options, x_train, y_train, x_test, y_test, x_val, y_val, y_pred_train, y_pred_test, y_pred_val, score)
    
    copyfile(sys.argv[0], path + os.path.basename(sys.argv[0]))
    try:
        copytree('moloi/models', path + 'models')
    except:  # FileNotFoundError not comp python2
        pass

    path_old = path[:-1]

    try:
        path = (path[:-8] + '_' + section +  '_' + str(descriptors) +  '_' + str(round(accuracy_test, 3)) +'/').replace(" ", "_")
        os.rename(path_old, path)
    except TypeError:
        pass

    descriptors = eval(descriptors)
    features = []
    for i in descriptors:
        if i == 'rdkit':
            features.append(list(rdkit_fetures_names()))
        if i == 'mordred':
            features.append(list(mordred_fetures_names()))
        if i == 'maccs':
            features.append(list("maccs_"+str(i) for i in range(167)))
        if i == 'morgan':
            features.append(list("morgan_"+str(i) for i in range(options.n_bits)))
        if i == 'spectrophore':
            features.append(list("spectrophore_"+str(i) for i in range(options.n_bits)))
    features = sum(features, [])

    if options.select_model in ['xgb','rf']:
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)
            indices = indices[-30:]
            
            plot_fi(indices, importances, features, path)
            
            importances = np.array(importances).reshape(-1,1)
            features = np.array(features).reshape(-1,1)

            tab = np.hstack([features, importances])

            fi = pd.DataFrame(tab)

            fi.to_csv(path+"feature_importance.csv", sep=",", header=["feature","importance"], index=False)
        except:
            pass
    else:
        try:
            importances = []
            X = list(x_test)
            for i in range(x_test.shape[1]):
                x_test = np.array(list(X[:]))
                x = m_mean(x_test, i)
                test_proba = model.predict_proba(x)
                try:
                    auc = roc_auc_score(y_test, test_proba[:,1])
                except:
                    auc = roc_auc_score(y_test, test_proba)

                importances.append(auc_test-auc)

            indices = np.argsort(importances)
            indices = indices[-30:]
            x_label = 'AUC ROC test - AUC ROC without feature'
            
            plot_fi(indices, importances, features, path, x_label)
            
            importances = np.array(importances).reshape(-1,1)
            features = np.array(features).reshape(-1,1)

            tab = np.hstack([features, importances])

            fi = pd.DataFrame(tab)

            fi.to_csv(path+"feature_importance.csv", sep=",", header=["feature","importance"], index=False)
        except:
            pass
    
    #plot_TSNE(x_train, y_train, path)
    logger.info("Results path: %s", path)

    return accuracy_test, accuracy_train, rec, auc_test, auc_val, f1, path
