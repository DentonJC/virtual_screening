import argparse
import os
from datetime import datetime


def get_options():
    parser = argparse.ArgumentParser(prog="model data section")
    parser.add_argument('--select_model', help='name of the model, select from list in README')
    parser.add_argument('--data_config', help='path to dataset config file')
    parser.add_argument('--section', help='name of section in model config file')
    parser.add_argument('--load_model', help='path to model .sav')
    parser.add_argument('--descriptors', type=str, default=['mordred', 'maccs'], help='descriptor of molecules')
    parser.add_argument('--output', default=os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") +
                        "/tmp/" + str(datetime.now()) + '/', help='path to output directory')
    parser.add_argument('--model_config', default="/data/model_configs/bace.ini", help='path to config file')
    parser.add_argument('--n_bits', default=256, type=int, help='number of bits in Morgan fingerprint')
    parser.add_argument('--n_cv', default=5, type=int, help='number of splits in RandomizedSearchCV')
    parser.add_argument('--n_iter', default=6, type=int, help='number of iterations in RandomizedSearchCV')
    parser.add_argument('--n_jobs', default=-1, type=int, help='number of jobs')
    parser.add_argument('--patience', '-p', default=100, type=int, help='patience of fit')
    parser.add_argument('--gridsearch', '-g', action='store_true', default=False, help='use gridsearch')
    parser.add_argument('--metric', default='accuracy', choices=['accuracy', 'roc_auc', 'f1', 'matthews'],
                        help='metric for RandomizedSearchCV')
    parser.add_argument('--split_type', choices=['stratified', 'scaffold', 'random', 'cluster'], default='stratified',
                        type=str, help='type of train-test split')
    parser.add_argument('--split_s', default=0.2, type=float, help='size of test and valid splits')
    parser.add_argument('--targets', '-t', type=list, default=[0], help='set number of target column')
    parser.add_argument('--experiments_file', '-e', default='experiments.csv',
                        help='where to write results of experiments')
    return parser
