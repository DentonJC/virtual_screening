#!/usr/bin/env python

import os
import sys
import socket
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from sklearn.metrics import roc_auc_score, roc_curve


def create_report(path, accuracy_test, accuracy_train, rec, auc, f1, timer, rparams, tstart, history, random_state, options, x_train, y_train, x_test, y_test):
    """
    Create .pdf with information about experiment.
    """
    report_name = path+"report "+str(round(accuracy_test, 2))+".pdf"
    doc = SimpleDocTemplate(report_name, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    Report = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    
    string = str(rparams)
    string = string.replace("{", "")
    string = string.replace("'", "")
    string = string.replace("}", "")
    string = string.replace("\"", "")

    cmd = str(sys.argv)
    cmd = cmd.replace("[", "")
    cmd = cmd.replace("]", "")
    cmd = cmd.replace(",", " ")
    cmd = cmd.replace("'", "")

    ptext = '<font size=12> Command line input: %s </font>' % (cmd)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Arguments: %s </font>' % (str(options))
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Parameters: %s </font>' % (string)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Random state: %s </font>' % (str(random_state))
    Report.append(Paragraph(ptext, styles["Justify"]))
    
    ptext = '<font size=12> X_train shape: %s</font>' % str(x_train.shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Y_train shape: %s</font>' % str(y_train.shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> X_test shape: %s</font>' % str(x_test.shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Y_test shape: %s</font>' % str(y_test.shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    
    ptext = '<font size=12> Accuracy test: %s </font>' % (accuracy_test)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Accuracy train: %s </font>' % (accuracy_train)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Recall: %s </font>' % (rec)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> ROC AUC score: %s </font>' % (auc)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> f1 score: %s </font>' % (f1)
    Report.append(Paragraph(ptext, styles["Justify"]))
    
    ptext = '<font size=12> Started at: %s </font>' % (tstart)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Time required: %s </font>' % (timer)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Host name: %s </font>' % (socket.gethostname())
    Report.append(Paragraph(ptext, styles["Justify"]))
    Report.append(Spacer(1, 12))
    try:
        draw_history(history, path)
        im = Image(path+'history.png')
        Report.append(im)
    except:
        print("Can't create history plot for this type of experiment")
    if os.path.exists(path+'auc.png'):
        im = Image(path+'auc.png')
        Report.append(im)
    else:
        print("Can't append AUC plot for this type of experiment")

    doc.build(Report)
    print("Report complete, you can see it in the results folder")

def draw_history(history, path):
    """
    Create plot of history and save in path.
    """
    plt.figure(1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
    plt.subplot(211)

    keys = list(history.history.keys())

    plt.plot(history.history[keys[3]], color='r')
    plt.plot(history.history[keys[1]], color='g')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history[keys[2]], color='r')
    plt.plot(history.history[keys[0]], color='g')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.savefig(path+'history.png')
    plt.clf()


def AUC(pred_train, pred_test, pred_val, y_train, y_test, y_val, path):
    """
    https://www.wildcardconsulting.dk/useful-information/a-deep-tox21-neural-network-with-rdkit-and-keras/
    """
    auc_train = roc_auc_score(y_train, pred_train)
    try:
        auc_val = roc_auc_score(y_val, pred_val)
    except:
        pass
    auc_test = roc_auc_score(y_test, pred_test)
 
    fpr_train, tpr_train, _ =roc_curve(y_train, pred_train, pos_label=1)
    try:
        fpr_val, tpr_val, _ = roc_curve(y_val, pred_val, pos_label=1)
    except:
        pass
    fpr_test, tpr_test, _ = roc_curve(y_test, pred_test, pos_label=1)
 
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
    plt.savefig(path+'auc.png')
    
    
def plot_grid_search_2(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    """
    https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv#37163377
    """
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.savefig(path+'grid.png')


def plot_grid_search_3(score, k0, k1, k2):
    plt = sns.factorplot(x=k1, y='mean_test_score', col=k2, hue=k0, data=score);
    plt.savefig(path+'grid.png')


if __name__ == "__main__":
    # Create dummy report.
    path = os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/tmp/"
    accuracy_test = 0
    accuracy_train = 0
    rec_score = auc_score = f1_score = [0,0]
    timer = 0
    rparams = {"weights": "distance", "p": "2", "leaf_size": "30", "algorithm": "brute"}
    tstart = 0
    history = None
    random_state = 0
    options = [0, 0, 0]
    create_report(path, accuracy_test, accuracy_train, rec_score, auc_score, f1_score, timer, rparams, tstart, history, random_state, options)

    ### Добавить историю из загруженной модели траем и потестить plot_history
    pred_train = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
    pred_test = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
    pred_val = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
    y_train = [1, 1, 1, 1, 0, 0, 0, 1, 0, 1]
    y_test = [1, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    y_val = [1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    
    AUC(pred_train, pred_test, pred_val, y_train, y_test, y_val, path)
