#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_history(history, path):
    """
    Create plot of model fitting history and save in path.
    """
    keys = list(history.history.keys())
    
    plt.figure(figsize=(10, 4*len(keys)))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)
    
    colors = ['r','g','b']
    for i in range(len(keys)):
        plt.subplot(len(keys)*100 + 10 + i + 1)
        plt.plot(history.history[keys[i]], color=colors[i%len(colors)])
        plt.title(str(keys[i]))
        #plt.ylabel(str(keys[i]))
        #plt.xlabel('epoch')
    plt.savefig(path+'img/history.png')
    plt.clf()
    plt.cla()
    plt.close()


def plot_auc(x_train, x_test, x_val, y_train, y_test, y_val, path, train_proba, test_proba, val_proba, auc_train, auc_test, auc_val):
    """
    https://www.wildcardconsulting.dk/useful-information/a-deep-tox21-neural-network-with-rdkit-and-keras/
    """ 
    try:
        train_proba = train_proba[:,1]
        val_proba = val_proba[:,1]
        test_proba = test_proba[:,1]
    except:
        pass
    
    fpr_train, tpr_train, _ =roc_curve(y_train, train_proba, pos_label=1)
    try:
        fpr_val, tpr_val, _ = roc_curve(y_val, val_proba, pos_label=1)
    except:
        pass
    fpr_test, tpr_test, _ = roc_curve(y_test, test_proba, pos_label=1)

    plt.figure()
    lw = 2
    
    plt.plot(fpr_train, tpr_train, color='b',lw=lw, label='Train ROC (area = %0.2f)'%auc_train)
    try:
        plt.plot(fpr_val, tpr_val, color='g',lw=lw, label='Val ROC (area = %0.2f)'%auc_val)
    except:
        pass
    plt.plot(fpr_test, tpr_test, color='r',lw=lw, label='Test ROC (area = %0.2f)'%auc_test)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(path+'img/auc.png', dpi=100)
    plt.clf()
    plt.cla()
    plt.close()


def plot_grid_search(score, path):
    headers = list(score)
    columns = []
    for h in headers:
        if "param_" in h:
            columns.append(h)
    table = pd.DataFrame(score, columns=columns)

    for i, c in enumerate(columns):
        try:
            fig, ax = plt.subplots()
            plt.ylim([0.0, 1.0])
            plt.yticks(np.arange(0, 1, 0.1))
            plt.grid(True)
            plt.bar([str(i) for i in table[c]] , score["mean_test_score"], label=c)
            plt.xticks(rotation=45)
            plt.ylabel("mean_test_score")
            plt.title(c)
            fig.tight_layout()
        except ValueError:
            pass

        plt.savefig(path+'img/grid_'+c+'.png', dpi=1000)
        plt.clf()
        plt.cla()
        plt.close()


def plot_TSNE(x, y, path, title="t-SNE", label_1="inactive", label_2="active", n_components=2):
    print("t-SNE fitting")
    tsne = TSNE(n_components=n_components)
    coordinates = tsne.fit_transform(x)
    
    plt.scatter(coordinates[np.where(y==0)[0],0],
            coordinates[np.where(y==0)[0],1],
            c="red", s=5, alpha=0.4, label=label_1)
    plt.scatter(coordinates[np.where(y==1)[0],0],
            coordinates[np.where(y==1)[0],1],
            c="blue", s=5, alpha=0.4, label=label_2)

    plt.title(title)
    plt.legend()
    plt.savefig(path, dpi=1000)
    plt.clf()
    plt.cla()
    plt.close()
    

def plot_PCA(x, y, path, title="PCA", label_1="inactive", label_2="active", n_components=2):
    print("PCA fitting")
    pca = PCA(n_components=n_components)
    coordinates = pca.fit_transform(x)
    
    plt.scatter(coordinates[np.where(y==0)[0],0],
            coordinates[np.where(y==0)[0],1],
            c="red", s=5, alpha=0.4, label=label_1)
    plt.scatter(coordinates[np.where(y==1)[0],0],
            coordinates[np.where(y==1)[0],1],
            c="blue", s=5, alpha=0.4, label=label_2)

    plt.title(title)
    plt.legend()
    plt.savefig(path, dpi=1000)
    plt.clf()
    plt.cla()
    plt.close()