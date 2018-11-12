#!/usr/bin/env python

"""
An additional script to run a series of experiments described in table like
etc/experiments.csv where columns are hyperparameters and rows are experiments.
"""

# TODO: fix docstrings
import os
import matplotlib
matplotlib.use('Agg')  # plot without a running X server
import math
import random
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from joblib.parallel import BACKENDS
from moloi.moloi import experiment
from moloi.dictionaries import results_dict
from moloi.results import process_results


BACKEND = 'loki'
if BACKEND not in BACKENDS.keys():
    BACKEND = 'multiprocessing'

def isnan(x):
    """ Checking if the variable is NaN and type float. """
    return isinstance(x, float) and math.isnan(x)

def worker(i, table, exp_settings, common_gridsearch, result_cols, keys, params):
    """
    Check the rows of experiments_file in a loop. If there are no results in the row
    (empty fields after len(cols)), it takes all values in this row and calls the experiment
    function until all result fields are filled with step len(result_cols).

    Params
    ------
    params: list
        the names of the columns of not positional arguments in experiments_file
    keys: list
        not positional arguments of experiment function (run_model.py) in the same order as params
    result_cols: list
        the result metrics which will be added to experiments_file
        [accuracy_test, accuracy_train, rec, auc, auc_val, f1, gparams]
    common_gridsearch: bool
        one gridsearch for all experiments in row
    random_state: int
        random state of all
    experiments_file: string
        path to experiments table
    """

    rparams = False
    command = []
    for c, p in enumerate(params):
        if not isnan(table[p][i]):
            if keys[c] in ["-g", "--dummy"]:
                command.append(keys[c])
            elif keys[c] in ["--n_bits", "--n_cv", "--n_jobs", "-p", "--n_iter"]:
                command.append(keys[c])
                command.append(int(table[p][i]))
            elif keys[c] in ["--split"]:
                command.append(keys[c])
                command.append(float(table[p][i]))
            else:
                command.append(keys[c])
                command.append(str(table[p][i]))

    command.append("-e")
    command.append(experiments_file)
    for j in range(int(((table.shape[1] - len(params)) / len(result_cols)))):
        results = results_dict()
        final_command = command + ["-t"] + [int(j)]
        print(final_command)
        if isnan(table.iloc[i, j*len(result_cols) + len(params)]):
            if not common_gridsearch:
                rparams = False
            exp_settings['rparams'] = rparams
            results = experiment(final_command, exp_settings, results)
            model_address = results["model_address"]
            results["gparams"] = results["rparams"]

            # round results
            for key in results:
                if isinstance(results[key], (str, dict)):
                    continue
                elif isinstance(results[key], (list, np.ndarray)):
                    for p in range(len(results[key])):
                        results[key][p] = round(results[key][p], 4)
                else:
                    results[key] = round(results[key], 4)

            table = pd.read_csv(experiments_file)
            for p, r in enumerate(result_cols):
                table.iloc[i, j * len(result_cols) + len(params) + p] = str(results[r])

            if model_address:
                table.iloc[i, 4] = str(model_address)  # set on Load model column
                if "--load_model" not in command and common_gridsearch:
                    command.append("--load_model")
                    command.append(model_address)

            table.to_csv(experiments_file, index=False)


def main(experiments_file, common_gridsearch, random_state, result_cols, keys, params, verbose, n_jobs, refit, plots, callbacks):
    """ Process inputs and start workers. """
    logger_flag = False
    if not random_state and not isinstance(random_state, int):
        random_state = random.randint(1, 100)
    np.random.seed(random_state)
    # tf.set_random_seed(random_state)
    table = pd.read_csv(experiments_file)
    if n_jobs > table.shape[0]:
        n_jobs = table.shape[0] - 1

    exp_settings = {
        'experiments_file': os.path.basename(experiments_file).split('.')[0],
        'random_state': random_state,
        'rparams': False,
        'verbose': verbose,
        'refit': refit,
        'plots': plots,
        'callbacks': callbacks
        }

    Parallel(n_jobs=n_jobs, backend=BACKEND, verbose=verbose)(delayed(worker)(i, table, exp_settings, common_gridsearch, result_cols, keys, params) for i in range(table.shape[0]))


if __name__ == "__main__":
    keys = ["--load_model", "--output", "--model_config", "--descriptors", "--n_bits", "--n_cv",
            "--n_jobs", "-p", "-g", "--n_iter", "--metric", "--split_type", "--split_s",
            '--select_model', '--data_config']
    params = ["Load model", "Output", "Model config", "Descriptors", "n_bits", "n_cv", "n_jobs", "Patience",
              "Gridsearch", "n_iter", "Metric", "Split type", "Split size", 'Model', 'Data config']
    result_cols = ['balanced_accuracy_test', 'auc_test', 'auc_val']
    # result_cols = ['r2_test', 'r2_val', 'mae_test', 'mae_val']
    # plots = ["history", "AUC", "gridsearch", "feature_importance", "feature_importance_full", "results", "TSNE", "PCA", "correlation", "distributions"]
    plots = ["history", "AUC", "gridsearch", "results", "TSNE"]
    callbacks = "stopping, csv_logger, checkpoint"
    common_gridsearch = True
    random_state = 1337
    experiments_file = 'etc/test.csv'
    verbose = 0
    refit = False
    n_jobs = 1  # multiprocessing.cpu_count() # only for evaluation

    descriptors = [['rdkit', 'morgan', 'mordred', 'maccs'], ['rdkit'], ['morgan'], ['mordred'], ['maccs']]
    splits = ['cluster', 'scaffold', 'random', 'stratified']

    main(experiments_file, common_gridsearch, random_state, result_cols, keys, params, verbose, n_jobs, refit, plots, callbacks)
    
    filenames = ["maccs", "rdkit", "mordred", "morgan", "rdkit_maccs", "rdkit_mordred", "morgan_maccs", "morgan_mordred", "rdkit_morgan", "mordred_maccs", "rdkit_morgan_mordred_maccs"]
    titles = ['MACCS', 'RDKit', 'Mordred', 'Morgan', 'MACCS+RDKit', 'RDKit+Mordred', 'Morgan+MACCS', 'Morgan+Mordred', 'RDKit+Morgan', 'Mordred+MACCS', 'RDKit+Morgan+Mordred+MACCS']
    # NAME = ["clintox_scaffold", "clintox_random", "clintox_cluster", "bace_scaffold", "bace_random", "bace_cluster"]
    process_results(filenames, titles, [os.path.basename(experiments_file).split('.')[0]])
