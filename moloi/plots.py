#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pylab import *

params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': [8, 4] # 4.5, 4.5
    }
rcParams.update(params)


def plot_old_history(history, path):
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


def plot_history(history, path):
    columns = ['acc', 'loss']
    keys = list(history.history.keys())

    plt.figure(figsize=(10, 4*len(columns)))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)

    for i in range(2):
        plt.subplot(len(columns)*100 + 10 + i + 1)
        plt.plot(np.array(history.history[keys[i]]), label='train', color='b')
        plt.plot(np.array(history.history[keys[i+2]]), label='val', color='g')
        plt.title(str(columns[i]))
        plt.legend()
    plt.savefig(path+'img/history.png', dpi=150)
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
    plt.savefig(path+'img/auc.png', dpi=150)
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

        plt.savefig(path+'img/grid_'+c+'.png', dpi=150)
        plt.clf()
        plt.cla()
        plt.close()


"""
def plot_fi(indices, importances, features, path, x_label='Relative Importance'):
    fig = plt.figure(figsize=(8, 8),dpi=500)
    ax = fig.add_subplot(111)
    fig.tight_layout()
    plt.subplots_adjust(left=0.3, bottom=0.1, right=0.9, top=0.9)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), [importances[i] for i in indices], color='b', alpha=1, align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])

    plt.xlabel(x_label)
    plt.savefig(path+"img/feature_importance.png", dpi=500)
    plt.clf()
    plt.cla()
    plt.close()
"""
import matplotlib.patches as mpatches
from moloi.descriptors.rdkit import rdkit_fetures_names
from moloi.descriptors.mordred import mordred_fetures_names

def col(i):
    if 'morgan' in i:
        return 'r'
    if i in mordred_fetures_names():
        return 'g'
    if i in rdkit_fetures_names():
        return 'b'
    if 'maccs' in i:
        return 'y'
    else:
        return 'm'

def plot_fi(indices, importances, features, path, x_label='Relative Importance'):
    fig = plt.figure(figsize=(8, 8),dpi=150)
    ax = fig.add_subplot(111)
    fig.tight_layout()
    plt.subplots_adjust(left=0.3, bottom=0.1, right=0.9, top=0.9)
    s = pd.Series([importances[i] for i in indices], index=[features[i] for i in indices])
    plt.title('Feature Importances')
    
    try:
        color = [list([col(features[i]) for i in indices])]
        s.plot(kind='barh', color=color, alpha=0.6)
    except:
        color = [col(features[i]) for i in indices]
        s.plot(kind='barh', color=color, alpha=0.6)
    
    r = mpatches.Patch(color='r', label='Morgan', alpha=0.6)
    g = mpatches.Patch(color='g', label='mordred', alpha=0.6)
    b = mpatches.Patch(color='b', label='RDKit', alpha=0.6)
    y = mpatches.Patch(color='y', label='MACCS', alpha=0.6)
    plt.legend(handles=[r,g,b,y])

    plt.xlabel(x_label)
    plt.savefig(path, dpi=150)
    plt.clf()
    plt.cla()
    plt.close()


def plot_TSNE(x, y, y_a, path, titles, label_1, label_2, label_3, c1='r', c2='b', c3='#00FF00', s=2, alpha=1, n_components=2):    
    print("t-SNE fitting")
    tsne = TSNE(n_components=n_components)
    coordinates = tsne.fit_transform(x)

    fig = plt.figure()#(figsize=(3, 6.2),dpi=150)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.90, wspace = 0.2)
    ax1 = fig.add_subplot(121)
    ax1.scatter(coordinates[np.where(y==1)[0],0],
            coordinates[np.where(y==1)[0],1],
            c=c1, s=s, alpha=alpha, label=label_1[0])
    ax1.scatter(coordinates[np.where(y==0)[0],0],
            coordinates[np.where(y==0)[0],1],
            c=c2, s=s, alpha=alpha, label=label_2[0])
    
    if len(np.unique(y)) == 3:
        ax1.scatter(coordinates[np.where(y==2)[0],0],
                coordinates[np.where(y==2)[0],1],
                c=c3, s=s, alpha=alpha, label=label_3[0])

    ax1.set_title(titles[0])
    ax1.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.scatter(coordinates[np.where(y_a==1)[0],0],
            coordinates[np.where(y_a==1)[0],1],
            c=c1, s=s, alpha=alpha, label=label_1[1])
    ax2.scatter(coordinates[np.where(y_a==0)[0],0],
            coordinates[np.where(y_a==0)[0],1],
            c=c2, s=s, alpha=alpha, label=label_2[1])
    
    if len(np.unique(y_a)) == 3:
        ax2.scatter(coordinates[np.where(y_a==2)[0],0],
                coordinates[np.where(y_a==2)[0],1],
                c=c3, s=s, alpha=alpha, label=label_3[1])

    ax2.set_title(titles[1])
    ax2.legend()
    fig.savefig(path)#, dpi=150)
    fig.clf()


def plot_PCA(x, y, y_a, path, titles, label_1, label_2, label_3, c1='r', c2='b', c3='#00FF00', s=2, alpha=1, n_components=2):
    print("PCA fitting")
    pca = PCA(n_components=n_components)
    coordinates = pca.fit_transform(x)

    fig = plt.figure()#(figsize=(3, 6.2),dpi=150)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.90, wspace = 0.2)
    ax1 = fig.add_subplot(121)
    ax1.scatter(coordinates[np.where(y==1)[0],0],
            coordinates[np.where(y==1)[0],1],
            c=c1, s=s, alpha=alpha, label=label_1[0])
    ax1.scatter(coordinates[np.where(y==0)[0],0],
            coordinates[np.where(y==0)[0],1],
            c=c2, s=s, alpha=alpha, label=label_2[0])
    
    if len(np.unique(y)) == 3:
        ax1.scatter(coordinates[np.where(y==2)[0],0],
                coordinates[np.where(y==2)[0],1],
                c=c3, s=s, alpha=alpha, label=label_3[0])

    ax1.set_title(titles[0])
    ax1.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.scatter(coordinates[np.where(y_a==1)[0],0],
            coordinates[np.where(y_a==1)[0],1],
            c=c1, s=s, alpha=alpha, label=label_1[1])
    ax2.scatter(coordinates[np.where(y_a==0)[0],0],
            coordinates[np.where(y_a==0)[0],1],
            c=c2, s=s, alpha=alpha, label=label_2[1])
    
    if len(np.unique(y_a)) == 3:
        ax2.scatter(coordinates[np.where(y_a==2)[0],0],
                coordinates[np.where(y_a==2)[0],1],
                c=c3, s=s, alpha=alpha, label=label_3[1])

    ax2.set_title(titles[1])
    ax2.legend()
    fig.savefig(path)#, dpi=150)
    fig.clf()
