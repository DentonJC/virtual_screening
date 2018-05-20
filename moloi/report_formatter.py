#!/usr/bin/env python

import os
import sys
import socket
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_report(logger, path, accuracy_test, accuracy_train, rec, auc_train, auc_test, auc_val, train_proba, test_proba, val_proba, f1, timer, rparams, tstart, history, random_state, options, x_train, y_train, x_test, y_test, x_val, y_val, pred_train, pred_test, pred_val, score):
    """
    Create .pdf with information about experiment.
    """
    logger.info("Creating report")
    try:
        os.makedirs(path+"img/")
    except FileExistsError:
        pass
    report_path = os.path.join(path, "report " + str(round(accuracy_test, 2)) + ".pdf")
    doc = SimpleDocTemplate(report_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    Report = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    rparams = str(rparams)
    rparams = rparams.replace("{", "")
    rparams = rparams.replace("'", "")
    rparams = rparams.replace("}", "")
    rparams = rparams.replace("\"", "")

    cmd = str(sys.argv)
    cmd = cmd.replace("[", "")
    cmd = cmd.replace("]", "")
    cmd = cmd.replace(",", " ")
    cmd = cmd.replace("'", "")
    
    if type(options.n_cv) is not int:
        options.n_cv = "indices"
    options = str(options)
    options = options.replace("Namespace(", '<br />\n')
    options = options.replace(", ", '<br />\n')
    options = options.replace(",\t", '<br />\n')
    options = options.replace(")", "")
    

    ptext = '<font size=12> <b> Command line input: </b> %s </font>' % cmd
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Arguments: </b> %s </font>' % str(options)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Parameters: </b> %s </font>' % rparams
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Random state: </b> %s </font>' % str(random_state)
    Report.append(Paragraph(ptext, styles["Justify"]))

    ptext = '<font size=12> <b> X_train shape: </b> %s</font>' % str(x_train.shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Y_train shape: </b> %s</font>' % str(y_train.shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> X_val shape: </b> %s</font>' % str(x_val.shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Y_val shape: </b> %s</font>' % str(y_val.shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> X_test shape: </b> %s</font>' % str(x_test.shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Y_test shape: </b> %s</font>' % str(y_test.shape)
    Report.append(Paragraph(ptext, styles["Justify"]))

    ptext = '<font size=12> <b> Accuracy test: </b> %s </font>' % accuracy_test
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Accuracy train: </b> %s </font>' % accuracy_train
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Recall: </b> %s </font>' % rec
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> ROC AUC score: </b> %s </font>' % auc_test
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> f1 score: </b> %s </font>' % f1
    Report.append(Paragraph(ptext, styles["Justify"]))

    ptext = '<font size=12> <b> Started at: </b> %s </font>' % tstart
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Time required: </b> %s </font>' % timer
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Host name: </b> %s </font>' % socket.gethostname()
    Report.append(Paragraph(ptext, styles["Justify"]))
    Report.append(Spacer(1, 12))
    
    try:
        plot_history(history, path)
    except:
        logger.info("Can not create history plot for this experiment")


    try:
        plot_auc(x_train, x_test, x_val, y_train, y_test, y_val, path, train_proba, test_proba, val_proba, auc_train, auc_test, auc_val)
        im = Image(path+'img/auc.png')
        Report.append(im)
    except:
        logger.info("Can't plot ROC AUC for this experiment")
        pass

    
    try:
        plot_grid_search(score, path)
    except:
        logger.info("Can't plot gridsearch for this experiment")

    # Too slow!
    # concat train, test, val
    # try:
    #     plot_TSNE(x_train, y_train, path)
    # except TypeError:
    #     y_train = [item for sublist in y_train for item in sublist]
    #     plot_TSNE(x_train, y_train, path)

    
    doc.build(Report)
    logger.info("Report complete, you can see it in the results folder")


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


def plot_TSNE(x, y, path, n_components=2):
    print("t-SNE fitting")
    tsne = TSNE(n_components=n_components)
    coordinates = tsne.fit_transform(x)
    
    plt.scatter(coordinates[np.where(y==0)[0],0],
            coordinates[np.where(y==0)[0],1],
            c="red", s=50, alpha=0.4, label="inactive")
    plt.scatter(coordinates[np.where(y==1)[0],0],
            coordinates[np.where(y==1)[0],1],
            c="blue", s=50, alpha=0.4, label="active")

    plt.title("t-SNE")
    plt.legend()
    plt.savefig(path+'img/t-SNE.png', dpi=1000)
    plt.clf()
    plt.cla()
    plt.close()
    

def plot_PCA(x, y, path, n_components=2):
    print("PCA fitting")
    pca = PCA(n_components=n_components)
    coordinates = pca.fit_transform(x)
    
    plt.scatter(coordinates[np.where(y==0)[0],0],
            coordinates[np.where(y==0)[0],1],
            c="red", s=50, alpha=0.4, label="inactive")
    plt.scatter(coordinates[np.where(y==1)[0],0],
            coordinates[np.where(y==1)[0],1],
            c="blue", s=50, alpha=0.4, label="active")

    plt.title("t-SNE")
    plt.legend()
    plt.savefig(path+'img/t-SNE.png', dpi=1000)
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == "__main__":
    """
    Create dummy report.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    path = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp"
    accuracy_test = 0
    accuracy_train = 0
    rec = auc = f1 = [0,0]
    timer = 0
    rparams = {"some": "params"}
    tstart = 0
    history = None
    random_state = 0
    options = [0, 0, 0]
    score = False
    
    pred_train = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 1])
    pred_test = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 1])
    pred_val = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 1])
    y_train = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    y_test = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 1])
    y_val = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    
    x_train = np.array([[1, 1, 1, 1, 0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 1]])
    x_test = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 1]])
    x_val = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0, 1]])
    
    create_report(logger, path, accuracy_test, accuracy_train, rec, auc, f1, timer, rparams, tstart, history, random_state, options, 
                x_train, y_train, x_test, y_test, x_val, y_val, pred_train, pred_test, pred_val, score)
