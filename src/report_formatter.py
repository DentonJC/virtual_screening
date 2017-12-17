#!/usr/bin/env python

import os
import sys
import socket
import matplotlib.pyplot as plt
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from sklearn.metrics import roc_auc_score, roc_curve


def create_report(path, score, timer, rparams, tstart, history, random_state, options):
    """
    Create .pdf with information about experiment.
    """
    report_name = path+"report "+str(round(score[1], 2))+".pdf"
    doc = SimpleDocTemplate(report_name, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    Report = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    
    try:
        rparams = rparams['params'][0]
    except KeyError:
        pass
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
    ptext = '<font size=12> Score: %1.3f </font>' % (score[0])
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Accuracy: %1.3f </font>' % (score[1])
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


def auc(model, X_train, X_test, X_val, y_train, y_test, y_val, path):
    """
    https://www.wildcardconsulting.dk/useful-information/a-deep-tox21-neural-network-with-rdkit-and-keras/
    """
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
 
    auc_train = roc_auc_score(y_train, pred_train)
    auc_val = roc_auc_score(y_val, pred_val)
    auc_test = roc_auc_score(y_test, pred_test)
 
    fpr_train, tpr_train, _ =roc_curve(y_train, pred_train, pos_label=1)
    fpr_val, tpr_val, _ = roc_curve(y_val, pred_val, pos_label=1)
    fpr_test, tpr_test, _ = roc_curve(y_test, pred_test, pos_label=1)
 
    plt.figure()
    lw = 2
    plt.plot(fpr_train, tpr_train, color='b',lw=lw, label='Train ROC (area = %0.2f)'%auc_train)
    plt.plot(fpr_val, tpr_val, color='g',lw=lw, label='Val ROC (area = %0.2f)'%auc_val)
    plt.plot(fpr_test, tpr_test, color='r',lw=lw, label='Test ROC (area = %0.2f)'%auc_test)
 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(path+'auc.png')


if __name__ == "__main__":
    # Create dummy report.
    path = os.path.dirname(os.path.realpath(__file__)).replace("/src", "") + "/tmp/"
    score = [0,0]
    timer = 0
    rparams = []
    tstart = 0
    history = None
    create_report(path, score, timer, rparams, tstart, history)
