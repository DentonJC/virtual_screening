#!/usr/bin/env python

import os
import pylab
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D # keep it
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from joblib.parallel import BACKENDS
from moloi.descriptors.rdkit_descriptor import rdkit_fetures_names
from moloi.descriptors.mordred_descriptor import mordred_fetures_names
from moloi.data_processing import m_mean
BACKEND = 'loki'
if BACKEND not in BACKENDS.keys():
    BACKEND = 'multiprocessing'
# mlp.use('Agg')


params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': [8, 4]  # 4.5, 4.5
    }
pylab.rcParams.update(params)


def plot_history(logger, history, path):
    logger.info("Creating history plot")
    try:
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
    except:
        logger.info("Can not create history plot for this experiment")


def plot_auc(logger, data, path, train_proba, test_proba, val_proba, auc_train, auc_test, auc_val):
    """
    https://www.wildcardconsulting.dk/useful-information/a-deep-tox21-neural-network-with-rdkit-and-keras/
    """
    logger.info("Creating ROC AUC plot")
    try:
        try:
            train_proba = train_proba[:, 1]
            val_proba = val_proba[:, 1]
            test_proba = test_proba[:, 1]
        except:
            pass

        fpr_train, tpr_train, _ = roc_curve(data["y_train"], train_proba, pos_label=1)
        try:
            fpr_val, tpr_val, _ = roc_curve(data["y_val"], val_proba, pos_label=1)
        except:
            pass
        fpr_test, tpr_test, _ = roc_curve(data["y_test"], test_proba, pos_label=1)

        plt.figure()
        lw = 2

        plt.plot(fpr_train, tpr_train, color='b', lw=lw, label='Train ROC (area = %0.2f)' % auc_train)
        try:
            plt.plot(fpr_val, tpr_val, color='g', lw=lw, label='Val ROC (area = %0.2f)' % auc_val)
        except:
            pass
        plt.plot(fpr_test, tpr_test, color='r', lw=lw, label='Test ROC (area = %0.2f)' % auc_test)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(path+'img/auc.png', dpi=80)
        plt.clf()
        plt.cla()
        plt.close()
    except:
        logger.info("Can't plot ROC AUC for this experiment")


def plot_grid_search(logger, score, path):
    logger.info("Creating GridSearch plot")
    try:
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
                names = []
                for i in table[c]:
                    if i in [None, False]:
                        names.append(str(i))
                    else:
                        names.append(i)

                names, values = zip(*sorted(zip(names, score["mean_test_score"])))
                values = np.array(values)
                set_red = False
                if len(np.unique(names)) > 2:
                    if values[0] == max(values):
                        set_red = str(names[0])
                    if values[-1] == max(values):
                        set_red = str(names[-1])

                names = [str(i) for i in table[c]]
                barlist = plt.bar(names, score["mean_test_score"], label=c, color='g', alpha=0.3)
                if set_red:
                    barlist[names.index(set_red)].set_color('r')
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
    except:
        logger.info("Can't plot gridsearch for this experiment")


def col(i):
    if 'morgan' in i:
        return 'r'
    if i in mordred_fetures_names():
        return 'g'
    if i in rdkit_fetures_names():
        return 'b'
    if 'maccs' in i:
        return 'y'
    return 'm'


def plot_fi(indices, importances, features, path, x_label='Relative Importance'):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    # ax = fig.add_subplot(111)
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
    plt.legend(handles=[r, g, b, y])

    plt.xlabel(x_label)
    plt.savefig(path, dpi=150)
    plt.clf()
    plt.cla()
    plt.close()


def plot_TSNE(x, y, y_a, path, titles, label_1, label_2, label_3, c1='r', c2='b', c3='#00FF00', s=2, alpha=1, n_components=2):
    print("t-SNE fitting")
    tsne = TSNE(n_components=n_components)
    coordinates = tsne.fit_transform(x)
    coords = pd.DataFrame(coordinates)
    coords.to_csv(path.replace('/img/', '/img/coordinates/').replace('.png', '.csv'))

    fig = plt.figure()  # (figsize=(3, 6.2),dpi=150)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.90, wspace=0.2)
    ax1 = fig.add_subplot(121)
    ax1.scatter(coordinates[np.where(y == 1)[0], 0],
                coordinates[np.where(y == 1)[0], 1],
                c=c1, s=s, alpha=alpha, label=label_1[0])
    ax1.scatter(coordinates[np.where(y == 0)[0], 0],
                coordinates[np.where(y == 0)[0], 1],
                c=c2, s=s, alpha=alpha, label=label_2[0])

    if len(np.unique(y)) == 3:
        ax1.scatter(coordinates[np.where(y == 2)[0], 0],
                    coordinates[np.where(y == 2)[0], 1],
                    c=c3, s=s, alpha=alpha, label=label_3[0])

    ax1.set_title(titles[0])
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.scatter(coordinates[np.where(y_a == 0)[0], 0],
                coordinates[np.where(y_a == 0)[0], 1],
                c=c1, s=s, alpha=alpha, label=label_1[1])
    ax2.scatter(coordinates[np.where(y_a == 1)[0], 0],
                coordinates[np.where(y_a == 1)[0], 1],
                c=c2, s=s, alpha=alpha, label=label_2[1])

    if len(np.unique(y_a)) == 3:
        ax2.scatter(coordinates[np.where(y_a == 2)[0], 0],
                    coordinates[np.where(y_a == 2)[0], 1],
                    c=c3, s=s, alpha=alpha, label=label_3[1])

    ax2.set_title(titles[1])
    ax2.legend()
    fig.savefig(path)  # , dpi=150)
    fig.clf()


def plot_result_TSNE(coordinates, y, y_a, path, title, label_1='correct', label_2='incorrect',
                     c1='r', c2='b', s=2, alpha=1):
    fig = plt.figure(figsize=(4, 4))
    coordinates = coordinates[1:, 1:]
    for i in range(len(coordinates)):
        if y_a[i] == y[i]:
            plt.scatter(x=(coordinates[i][0]), y=coordinates[i][1],
                        c=c1, s=3, alpha=alpha, label=label_1)
        else:
            plt.scatter(x=(coordinates[i][0]), y=coordinates[i][1],
                        c=c2, s=3, alpha=alpha, label=label_2)

    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    plt.title(title)
    r = mpatches.Patch(color=c1, label=label_1, alpha=0.6)
    b = mpatches.Patch(color=c2, label=label_2, alpha=0.6)
    plt.legend(handles=[r, b])
    plt.savefig(path)  # , dpi=150)
    plt.clf()

    # fig = plt.figure(figsize=(4, 4))
    for i in range(len(coordinates)):
        if(y_a[i] == y[i]) and (y[i] == 1):
            plt.scatter(x=(coordinates[i][0]), y=coordinates[i][1],
                        c=c1, s=3, alpha=alpha, label=label_1)
        if(y_a[i] != y[i]) and (y[i] == 1):
            plt.scatter(x=(coordinates[i][0]), y=coordinates[i][1],
                        c=c2, s=3, alpha=alpha, label=label_2)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title+' positive class')
    r = mpatches.Patch(color=c1, label=label_1, alpha=0.6)
    b = mpatches.Patch(color=c2, label=label_2, alpha=0.6)
    plt.legend(handles=[r, b])
    plt.savefig(path.replace('.png', '_pos.png'))  # , dpi=150)
    plt.clf()

    for i in range(len(coordinates)):
        if(y_a[i] == y[i]) and (y[i] == 0):
            plt.scatter(x=(coordinates[i][0]), y=coordinates[i][1],
                        c=c1, s=3, alpha=alpha, label=label_1)
        if(y_a[i] != y[i]) and (y[i] == 0):
            plt.scatter(x=(coordinates[i][0]), y=coordinates[i][1],
                        c=c2, s=3, alpha=alpha, label=label_2)

    plt.title(title+' negative class')
    r = mpatches.Patch(color=c1, label=label_1, alpha=0.6)
    b = mpatches.Patch(color=c2, label=label_2, alpha=0.6)
    plt.legend(handles=[r, b])
    plt.savefig(path.replace('.png', '_neg.png'))  # , dpi=150)
    plt.clf()


def plot_PCA(x, y, y_a, path, titles, label_1, label_2, label_3, c1='r', c2='b', c3='#00FF00',
             s=2, alpha=1, n_components=2):
    print("PCA fitting")
    pca = PCA(n_components=n_components)
    coordinates = pca.fit_transform(x)

    fig = plt.figure()  # (figsize=(3, 6.2),dpi=150)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.90, wspace=0.2)
    ax1 = fig.add_subplot(121)
    ax1.scatter(coordinates[np.where(y == 1)[0], 0],
                coordinates[np.where(y == 1)[0], 1],
                c=c1, s=s, alpha=alpha, label=label_1[0])
    ax1.scatter(coordinates[np.where(y == 0)[0], 0],
                coordinates[np.where(y == 0)[0], 1],
                c=c2, s=s, alpha=alpha, label=label_2[0])

    if len(np.unique(y)) == 3:
        ax1.scatter(coordinates[np.where(y == 2)[0], 0],
                    coordinates[np.where(y == 2)[0], 1],
                    c=c3, s=s, alpha=alpha, label=label_3[0])

    ax1.set_title(titles[0])
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.scatter(coordinates[np.where(y_a == 1)[0], 0],
                coordinates[np.where(y_a == 1)[0], 1],
                c=c1, s=s, alpha=alpha, label=label_1[1])
    ax2.scatter(coordinates[np.where(y_a == 0)[0], 0],
                coordinates[np.where(y_a == 0)[0], 1],
                c=c2, s=s, alpha=alpha, label=label_2[1])

    if len(np.unique(y_a)) == 3:
        ax2.scatter(coordinates[np.where(y_a == 2)[0], 0],
                    coordinates[np.where(y_a == 2)[0], 1],
                    c=c3, s=s, alpha=alpha, label=label_3[1])

    ax2.set_title(titles[1])
    ax2.legend()
    fig.savefig(path)  # , dpi=150)
    fig.clf()


def find_importances(i, X, y_test, model, auc_test):
    x_test = np.array(list(X[:]))
    x = m_mean(x_test, i)
    test_proba = model.predict_proba(x)
    try:
        auc = roc_auc_score(y_test, test_proba[:, 1])
    except:
        auc = roc_auc_score(y_test, test_proba)
    return auc_test-auc


def plot_features_importance(logger, options, data, model, path, auc_test):
    logger.info("Creating feature importance plot")
    features = []
    for i in options.descriptors:
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

    if options.select_model in ['none']:  # ['xgb','rf'] - but number of descriptors is too big
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)
            indices = indices[-30:]
            x_label = 'AUC ROC test - AUC ROC without feature'
            try:
                plot_fi(indices, importances, features, path, x_label)
            except:
                pass

            importances = np.array(importances).reshape(-1, 1)
            features = np.array(features).reshape(-1, 1)

            tab = np.hstack([features, importances])

            fi = pd.DataFrame(tab)

            fi.to_csv(path+"feature_importance.csv", sep=",", header=["feature", "importance"], index=False)
            logger.info("Feature importance plot created")
        except:
            logger.info("Can not plot feature importance")
    else:
        try:
            X = list(data["x_test"])

            importances = []
            importances.append(Parallel(n_jobs=options.n_jobs, backend=BACKEND, verbose=1)(delayed(find_importances)(i, X, data["y_test"], model, auc_test) for i in range(data["x_test"].shape[1])))
            importances = importances[0]
            indices = np.argsort(importances)
            x_label = 'AUC ROC test - AUC ROC without feature'

            try:
                plot_fi(indices[-30:], importances, features, path+"img/feature_importance.png", x_label)
            except:
                pass

            try:
                plot_fi(indices, importances, features, path+"img/feature_importance_full.png", x_label)
            except:
                pass

            importances = np.array(importances).reshape(-1, 1)
            features = np.array(features).reshape(-1, 1)

            tab = np.hstack([features, importances])

            fi = pd.DataFrame(tab)

            fi.to_csv(path+"feature_importance.csv", sep=",", header=["feature", "importance"], index=False)
            logger.info("Feature importance plot created")
        except:
            logger.info("Can not plot feature importance")


def plot_results(logger, options, data, y_pred_train, y_pred_test, y_pred_val, path):
    try:
        logger.info("Creating results plot")
        root_address = os.path.dirname(os.path.realpath(__file__)).replace('/moloi', '')
        addresses_tsne = root_address + "/etc/img/coordinates/" + options.data_config.replace('/data/data_configs/', '').replace('.ini', '').replace(root_address, '') + "/" + str(options.descriptors) + "/tsne/t-SNE_" + options.split_type + ".png"
        X_tsne = pd.read_csv(addresses_tsne.replace('.png', '.csv'), header=None)
        Y = np.c_[data["y_train"].T, data["y_test"].T]
        Y = np.c_[Y, data["y_val"].T]

        Y_a = [y_pred_train, y_pred_test, y_pred_val]
        Y_a = [i for sub in Y_a for i in sub]

        Y_a = np.array(Y_a)
        X_tsne = np.array(X_tsne)
        
        title_tsne = "t-SNE " + os.path.splitext(os.path.basename(options.data_config))[0] + " " + options.split_type + " result"

        s = 3
        alpha = 0.5
        plot_result_TSNE(X_tsne, Y.T, Y_a, path + "img/results.png", title=title_tsne, s=s, alpha=alpha)
        logger.info("Results plot created")
    except:
        logger.info("Can not plot results")


def plot_rep_TSNE(x, y, path, title="t-SNE", label_1="active", label_2="inactive", label_3="", c1='r', c2='b', c3='#00FF00', s=2, alpha=1, n_components=2):
    print("t-SNE fitting")
    tsne = TSNE(n_components=n_components)
    coordinates = tsne.fit_transform(x)

    plt.scatter(coordinates[np.where(y==1)[0],0],
            coordinates[np.where(y==1)[0],1],
            c=c1, s=s, alpha=alpha, label=label_1)
    plt.scatter(coordinates[np.where(y==0)[0],0],
            coordinates[np.where(y==0)[0],1],
            c=c2, s=s, alpha=alpha, label=label_2)
    
    if len(np.unique(y)) == 3:
        plt.scatter(coordinates[np.where(y==2)[0],0],
                coordinates[np.where(y==2)[0],1],
                c=c3, s=s, alpha=alpha, label=label_3)

    plt.title(title)
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.clf()
    plt.cla()
    plt.close()
    

def plot_rep_PCA(x, y, path, title="PCA", label_1="active", label_2="inactive", label_3="", c1='r', c2='b', c3='#00FF00', s=2, alpha=1, n_components=2):
    print("PCA fitting")
    pca = PCA(n_components=n_components)
    coordinates = pca.fit_transform(x)

    plt.scatter(coordinates[np.where(y==1)[0],0],
            coordinates[np.where(y==1)[0],1],
            c=c1, s=s, alpha=alpha, label=label_1)
    plt.scatter(coordinates[np.where(y==0)[0],0],
            coordinates[np.where(y==0)[0],1],
            c=c2, s=s, alpha=alpha, label=label_2)
    
    if len(np.unique(y)) == 3:
        plt.scatter(coordinates[np.where(y==2)[0],0],
                coordinates[np.where(y==2)[0],1],
                c=c3, s=s, alpha=alpha, label=label_3)

    plt.title(title)
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.clf()
    plt.cla()
    plt.close()
