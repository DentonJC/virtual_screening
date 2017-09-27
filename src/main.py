import os
import sys
import glob
import getopt
import logging
import numpy as np
import pandas as pd
from keras.optimizers import * # for compile_optimizer()
from shutil import copyfile, copytree
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from datetime import datetime
from src.report import create_report, draw_history
import configparser
config = configparser.ConfigParser()


def read_cmd():
    """ 
    Read and return the script arguments.
    """
    MACCS = False
    Morgan = True
    GRID_SEARCH = False
    DUMMY = False
    nBits = 1024
    patience = 100
    time_start = datetime.now()
    path = os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/tmp/" + str(time_start) + '/'
    config_path = os.path.dirname(os.path.realpath(__file__)) + "/configs/configs.ini"
    section = ''
    n_jobs = -1
    
    targets, features, set_targets, set_features = False, False, False, False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hifnpocsab:gd", ["help", "input=", "feature=", "nBits=", "patience=", "output=", "config=", "section=", "targets=", "features="])
    except getopt.GetoptError:
        print ("Usage : script.py -i <input_file> -c <config_file> -s <section> -f <featurizer> -n <number_of_bits> -o <output_file> -p <patience> -g (grid_search) -d (dummy_data) or \
                script.py --input <input_file> --config <config_file> --section <section> --feature <featurizer> --nBits <number_of_bits> --output <output_file> --patience <patience> -g (grid_search) -d (dummy_data)")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ("Usage : script.py -i <input_file> -c <config_file> -s <section> -f <featurizer> -n <number_of_bits> -o <output_file> -p <patience> -g (grid_search) -d (dummy_data) or \
                script.py --input <input_file> --config <config_file> --section <section> --feature <featurizer> --nBits <number_of_bits> --output <output_file> --patience <patience> -g (grid_search) -d (dummy_data)")
        if opt in ('-i', '--input'):
            data_file = arg
        if opt in ('-f', '--feature'):
            if arg in ("maccs", "MACCS"):
                MACCS = True
                Morgan = False
            else:
                MACCS = False
                Morgan = True
        if opt in ('-n', '--nBits'):
            nBits = arg
        if opt in ('-p', '--patience'):
            patience = arg
        if opt in ('-o', '--output'):
            if arg:
                path = arg + str(time_start) + '/'
        if opt in ('-c', '--config'):
            config_path = arg
        if opt in ('-s', '--section'):
            section = arg
            
        if opt in ('-a', '--targets'):
            targets = arg
        if opt in ('-b', '--features'):
            features = arg    
        
        if opt in ('-g'):
            GRID_SEARCH = True
        if opt in ('-d'):
            DUMMY = True
        
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path+"results/*"):
        os.makedirs(path+"results/")
        
    # filepath = path + "results/" + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    filepath = path + "results/" + "weights-improvement-{epoch:02d}.hdf5" # in case if metrics is not val_acc
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=eval(patience), verbose=0, mode='auto')
    csv_logger = CSVLogger(path + 'history_' + os.path.basename(sys.argv[0]).replace(".py", "") + 
                            "_" + os.path.basename(data_file).replace(".csv", "") + '.csv', append=True, separator=';')
    callbacks_list = [checkpoint, stopping, csv_logger]
    
    if targets:
        set_targets = [eval(targets)]
    if features:
        set_features = range(0,eval(features))

    return DUMMY, GRID_SEARCH, data_file, MACCS, Morgan, path, time_start, filepath, callbacks_list, config_path, section, int(nBits), set_targets, set_features, n_jobs


def read_config(config_path, section):
    config.read(config_path)
    def_config = config['DEFAULT']
    n_folds = eval(def_config['n_folds'])
    epochs = eval(def_config['epochs'])
    n_iter = eval(def_config['n_iter'])
    class_weight = eval(def_config['class_weight'])
    model_config = config[section]  
    rparams = eval(model_config['rparams'])
    gparams = eval(model_config['gparams'])
    return n_folds, epochs, rparams, gparams, n_iter, class_weight
    
    
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


def load_model(loaded_model_json):
    json_file = open(loaded_model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model
    

class Logger(object):
    """https://stackoverflow.com/questions/11325019/output-on-the-console-and-file-using-python"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()    


def compile_optimizer(optimizer, learning_rate=0.1, momentum=0.1):
    if optimizer == 'Adam':
        return Adam(lr=learning_rate)
    elif optimizer == 'Nadam':
        return Nadam(lr=learning_rate)
    elif optimizer == 'Adamax':
        return Adamax(lr=learning_rate)
    elif optimizer == 'RMSprop':
        return RMSprop(lr=learning_rate)
    elif optimizer == 'Adagrad':
        return Adagrad(lr=learning_rate)
    elif optimizer == 'Adadelta':
        return Adadelta(lr=learning_rate)
    else:
        return SGD(lr=learning_rate, momentum=momentum)
        

def drop_nan(x, y):
    """
    Remove rows with NaN in data and labes relevant to this rows.
    """
    _, targ = x.shape
    table = np.c_[x, y]
    table = pd.DataFrame(table)
    table = table.dropna(axis=0, how='any')
    table = np.array(table)
    x = table[:, 0:targ]
    y = table[:, targ:]
    return x, y
    
        
def evaluate(path, model, x_train, x_test, x_val, y_train, y_test, y_val, time_start, rparams, history):   
    model_json = model.to_json()
    with open(path+"model.json", "w") as json_file:
        json_file.write(model_json)
    if rparams.get("metrics") == ['accuracy']:
        copyfile(get_latest_file(path), path + "best_weights.h5")
        #model.load_weights(get_latest_file(path))
    
    # print and save model summary
    orig_stdout = sys.stdout
    f = open(path + 'model', 'w')
    sys.stdout = f
    logging.info(model.summary())
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()

    # evaluate
    score = model.evaluate(x_test, y_test, batch_size=rparams.get("batch_size", 32), verbose=1)
    print('Score: %1.3f' % score[0])
    logging.info('Score: %1.3f' % score[0])
    print('Accuracy: %1.3f' % score[1])
    logging.info('Accuracy: %1.3f' % score[1])
    
    # find how long the program was running
    tstop = datetime.now()
    timer = tstop - time_start
    print(timer)
    logging.info(timer)
    
    # create report, prediction and save script and all current models
    create_report(path, score, timer, rparams, time_start, history)
    copyfile(sys.argv[0], path + os.path.basename(sys.argv[0]))
    copytree('src/models', path + 'models')
    y_pred = model.predict(x_test)
    result = [np.argmax(i) for i in y_pred]
    save_labels(result, path + "y_pred.csv")

    print("Done")
    print("Results path",path)
    logging.info("Done")
