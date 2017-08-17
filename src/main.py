import os 
import sys
import glob
import numpy as np
from shutil import copyfile
from pybeep.pybeep import PyVibrate, PyBeep
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import *
from datetime import datetime
from src.report import create_report, draw_history

    
def firing(patience):
    tstart = datetime.now() # start timer
    
    pyname = sys.argv[0]
    
    path = os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/tmp/" + str(tstart) + '/' # create folders
    filename = os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/data/HIV.csv"
            
    MACCS = True
    Morgan = False
    GRID_SEARCH = False
    DUMMY = False
    
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    if len(sys.argv) > 2:
        if sys.argv[2] in ("maccs", "MACCS"):
            MACCS = True
            Morgan = False
        else:
            MACCS = False
            Morgan = True
    
    if len(sys.argv) > 3:
        if sys.argv[3]:
            path = sys.argv[3] + str(tstart) + '/'
    
    if len(sys.argv) > 4:
        GRID_SEARCH = sys.argv[4]
        if GRID_SEARCH in "False":
            GRID_SEARCH = False
    
    if len(sys.argv) > 5:    
        DUMMY = sys.argv[5]
        if DUMMY in "False":
            DUMMY = False
 
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
        
    return(pyname, DUMMY, GRID_SEARCH, filename, MACCS, Morgan, path, tstart, filepath, callbacks_list)


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

def evaluate_and_done(path, model, x_test, y_test, tstart, rparams, history, pyname):
    model_json = model.to_json()
    with open(path+"model.json", "w") as json_file:
        json_file.write(model_json)

    #copyfile(get_latest_file(path), path+"best_weights.h5")

    #model.load_weights(get_latest_file(path))

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
    timer = tstop - tstart
    print(timer)
    create_report(path, score, timer, rparams, pos_score, neg_score, tstart, history)
    copyfile(pyname, path+os.path.basename(pyname))

    print("Done")

    ##Signal
    while True:
        PyVibrate().beep()
        PyVibrate().beepn(3)
        PyBeep().beep()
        PyBeep().beepn(3)
