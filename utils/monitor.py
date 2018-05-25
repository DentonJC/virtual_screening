import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


root_address = os.path.dirname(os.path.realpath(os.getcwd())).replace("/moloi/utils", "")
path = root_address + "/tmp/"

def plot_history(path):
    columns = ['acc', 'loss']
    history = pd.read_csv(path, header=0, sep=';')
    history = np.array(history)
    x = np.array(history[:,0])
    history = history[:,1:]

    plt.figure(figsize=(10, 4*len(columns)))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)

    for i in range(2):
        plt.subplot(len(columns)*100 + 10 + i + 1)
        plt.plot(x, np.array(history[:,i]), label='train', color='b')
        plt.plot(x, np.array(history[:,i+2]), label='val', color='g')
        plt.title(str(columns[i]))
        plt.legend()
        #plt.ylabel(str(keys[i]))
        #plt.xlabel('epoch')
    #plt.clf()
    #plt.cla()
    #plt.close()
    plt.show()
    
def get_latest_file(path):
    all_dirs = os.listdir(path)
    for i, d in enumerate(all_dirs):
        all_dirs[i] = path+d
    latest_subdir = max(all_dirs, key=os.path.getmtime)
    return latest_subdir

while True:
    #history = get_latest_file(path)
    history = path+"2018-05-24 06:12:16.234735"
    history = history+'/history_run_FCNN.csv'
    if os.path.isfile(history):
        plot_history(history)
    time.sleep(5)
