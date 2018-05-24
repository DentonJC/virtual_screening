#!/usr/bin/env python

import os
import sys
import socket
import logging
import numpy as np
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from moloi.plots import plot_history, plot_auc, plot_grid_search, plot_TSNE, plot_PCA


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
    
    if options:
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

    try:
        plot_grid_search(score, path)
    except:
        logger.info("Can't plot gridsearch for this experiment")

    # X = np.c_[x_train.T, x_test.T]
    # X = np.c_[X, x_val.T]
    # Y = np.c_[y_train.T, y_test.T]
    # Y = np.c_[Y, y_val.T]

    # plot_TSNE(X.T, Y.T, path+'img/t-SNE.png')
    # plot_PCA(X.T, Y.T, path+'img/PCA.png')

    doc.build(Report)
    logger.info("Report complete, you can see it in the results folder")


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
