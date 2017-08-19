import os
import sys
import glob
import numpy as np
import getopt
from shutil import copyfile
from pybeep.pybeep import PyVibrate, PyBeep
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import *
from datetime import datetime
from src.report import create_report, draw_history

    
def firing(patience):
    time_start = datetime.now() # start timer
    script_address = sys.argv[0]
    
    path = os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/tmp/" + str(time_start) + '/' # create folders
    data_file = os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/data/HIV.csv"
            
    MACCS = True
    Morgan = False
    GRID_SEARCH = False
    DUMMY = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hifo:gd",["help", "input=", "feature=", "output=", "grid=", "dummy="])
    except getopt.GetoptError:
        print ("Usage : script.py -i <input_file> -f <featurizer> -o <output_file> -g <grid_search> -d <dummy_data> or \
                script.py --input <input_file> --feature <featurizer> --output <output_file> --grid <grid_search> --dummy <dummy_data>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ("Usage : script.py -i <input_file> -f <featurizer> -o <output_file> -g <grid_search> -d <dummy_data> or \
                script.py --input <input_file> --feature <featurizer> --output <output_file> --grid <grid_search> --dummy <dummy_data>")
        if opt in ('-i', '--input'):
            data_file = arg
        if opt in ('-f', '--feature'):
            if arg in ("maccs", "MACCS"):
                MACCS = True
                Morgan = False
            else:
                MACCS = False
                Morgan = True
        if opt in ('-o', '--output'):
            if arg:
                path = arg + str(time_start) + '/'
        if opt in ('-g', '--grid'):
            GRID_SEARCH = True
        if opt in ('-d', '--dummy'):
            DUMMY = True
 
    print("PATH",path)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path+"results/*"):
        os.makedirs(path+"results/")
        
    #filepath = path + "results/" + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    filepath = path + "results/" + "weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
    csv_logger = CSVLogger(path+'history.csv', append=True, separator=';')
    callbacks_list = [checkpoint, stopping, csv_logger]
        
    return(script_address, DUMMY, GRID_SEARCH, data_file, MACCS, Morgan, path, time_start, filepath, callbacks_list)


def get_latest_file(path):
    list_of_files = glob.iglob(path+"results/*")
    
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


def interp(optimizer, learning_rate):
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
        return SGD(lr=learning_rate)

def evaluate_and_done(path, model, x_test, y_test, time_start, rparams, history, script_address):
    model_json = model.to_json()
    with open(path+"model.json", "w") as json_file:
        json_file.write(model_json)

    #copyfile(get_latest_file(path), path+"best_weights.h5")

    #model.load_weights(get_latest_file(path))
    
    orig_stdout = sys.stdout
    f = open(path+'model', 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()

    score = model.evaluate(x_test, y_test, batch_size=rparams.get("batch_size", 32), verbose=1)
    print('Score: %1.3f' % score[0])
    print('Accuracy: %1.3f' % score[1])

    if y_test.shape[1] == 1:
        x_ones, x_zeros = [], []
        for i in range(len(x_test)):
            if y_test[i]:
                x_ones.append(x_test[i])
            else:
                x_zeros.append(x_test[i])
                
        y_ones = [1] * len(x_ones)
        y_zeros = [0] * len(x_zeros)
        x_ones = np.array(x_ones)
        y_ones = np.array(y_ones)
        x_zeros = np.array(x_zeros)
        y_zeros = np.array(y_zeros)
        print(x_ones.shape, y_ones.shape)
        pos_score = model.evaluate(x_ones, y_ones, batch_size=rparams.get("batch_size", 32), verbose=1)
        print('Score pos: %1.3f' % pos_score[0])
        print('Accuracy pos: %1.3f' % pos_score[1])

        neg_score = model.evaluate(x_zeros, y_zeros, batch_size=rparams.get("batch_size", 32), verbose=1)
        print('Score neg: %1.3f' % neg_score[0])
        print('Accuracy neg: %1.3f' % neg_score[1])
    else:
        pos_score = (0,0)
        neg_score = (0,0)
    
    tstop = datetime.now()
    timer = tstop - time_start
    print(timer)
    create_report(path, score, timer, rparams, pos_score, neg_score, time_start, history)
    copyfile(script_address, path+os.path.basename(script_address))

    print("Done")

    ##Signal
    while True:
        PyVibrate().beep()
        PyVibrate().beepn(3)
        PyBeep().beep()
        PyBeep().beepn(3)
