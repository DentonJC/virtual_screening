#!/usr/bin/env python

import os
import sys
import socket
import logging
import numpy as np
import pandas as pd
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.platypus import Table, PageBreak, FrameBreak
from reportlab.lib.units import inch
from moloi.plots import plot_history, plot_auc, plot_grid_search, correlation, distributions
from moloi.plots import plot_features_importance, plot_results, plot_rep_TSNE, plot_rep_PCA


def create_report(logger, path, train_proba, test_proba, val_proba, timer, rparams,
                  tstart, history, random_state, options, data, pred_train,
                  pred_test, pred_val, score, model, results, plots):
    """
    Create .pdf with information about experiment.
    """
    logger.info("Creating report")
    try:
        os.makedirs(path+"img/")
    except FileExistsError:
        pass
    report_path = os.path.join(path, "report " + str(round(results["accuracy_test"], 2)) + ".pdf")
    doc = SimpleDocTemplate(report_path, pagesize=letter, rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    

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
        if isinstance(options.n_cv, int):
            options.n_cv = "indices"
    string_options = str(options)
    string_options = string_options.replace("Namespace(", '<br />\n')
    string_options = string_options.replace(", ", '<br />\n')
    string_options = string_options.replace(",\t", '<br />\n')
    string_options = string_options.replace(")", "")
    string_options = string_options.replace("=", " = ")

    ptext = '<font size=12> <b> Command line input: </b> %s </font>' % cmd
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Arguments: </b> %s </font>' % str(string_options)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Parameters: </b> %s </font>' % rparams
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Random state: </b> %s </font>' % str(random_state)
    Report.append(Paragraph(ptext, styles["Justify"]))

    ptext = '<font size=12> <b> X_train shape: </b> %s</font>' % str(data["x_train"].shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Y_train shape: </b> %s</font>' % str(data["y_train"].shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> X_val shape: </b> %s</font>' % str(data["x_val"].shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Y_val shape: </b> %s</font>' % str(data["y_val"].shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> X_test shape: </b> %s</font>' % str(data["x_test"].shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Y_test shape: </b> %s</font>' % str(data["y_test"].shape)
    Report.append(Paragraph(ptext, styles["Justify"]))
    
    if results["accuracy_test"] is not False:
        ptext = '<font size=12> <b> Accuracy test: </b> %s </font>' % results["accuracy_test"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["accuracy_train"] is not False:
        ptext = '<font size=12> <b> Accuracy train: </b> %s </font>' % results["accuracy_train"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["accuracy_val"] is not False:
        ptext = '<font size=12> <b> Accuracy val: </b> %s </font>' % results["accuracy_val"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["rec_test"] is not False:
        ptext = '<font size=12> <b> Recall test: </b> %s </font>' % results["rec_test"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["rec_train"] is not False:
        ptext = '<font size=12> <b> Recall train: </b> %s </font>' % results["rec_train"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["rec_val"] is not False:
        ptext = '<font size=12> <b> Recall val: </b> %s </font>' % results["rec_val"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["auc_test"] is not False:
        ptext = '<font size=12> <b> AUC test: </b> %s </font>' % results["auc_test"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["auc_train"] is not False:
        ptext = '<font size=12> <b> AUC train: </b> %s </font>' % results["auc_train"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["auc_val"] is not False:
        ptext = '<font size=12> <b> AUC val: </b> %s </font>' % results["auc_val"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["f1_test"] is not False:
        ptext = '<font size=12> <b> F1 test: </b> %s </font>' % results["f1_test"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["f1_train"] is not False:
        ptext = '<font size=12> <b> F1 train: </b> %s </font>' % results["f1_train"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["f1_val"] is not False:
        ptext = '<font size=12> <b> F1 val: </b> %s </font>' % results["f1_val"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["rmse_test"] is not False:
        ptext = '<font size=12> <b> RMSE test: </b> %s </font>' % results["rmse_test"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["rmse_train"] is not False:
        ptext = '<font size=12> <b> RMSE train: </b> %s </font>' % results["rmse_train"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["rmse_val"] is not False:
        ptext = '<font size=12> <b> RMSE val: </b> %s </font>' % results["rmse_val"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["mae_test"] is not False:
        ptext = '<font size=12> <b> MAE test: </b> %s </font>' % results["mae_test"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["mae_train"] is not False:
        ptext = '<font size=12> <b> MAE train: </b> %s </font>' % results["mae_train"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["mae_val"] is not False:
        ptext = '<font size=12> <b> MAE val: </b> %s </font>' % results["mae_val"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["r2_test"] is not False:
        ptext = '<font size=12> <b> R2 test: </b> %s </font>' % results["r2_test"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["r2_train"] is not False:
        ptext = '<font size=12> <b> R2 train: </b> %s </font>' % results["r2_train"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    if results["r2_val"] is not False:
        ptext = '<font size=12> <b> R2 val: </b> %s </font>' % results["r2_val"]
        Report.append(Paragraph(ptext, styles["Justify"]))
    
    ptext = '<font size=12> <b> Started at: </b> %s </font>' % tstart
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Time required: </b> %s </font>' % timer
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> <b> Host name: </b> %s </font>' % socket.gethostname()
    Report.append(Paragraph(ptext, styles["Justify"]))
    Report.append(Spacer(1, 12))
    
    # Plots
    if "history" in plots:
        plot_history(logger, path)
        if os.path.isfile(path+'img/history.png'):
            im = Image(path+'img/history.png', 8 * inch, 6 * inch, kind='proportional')
            Report.append(im)

    if "AUC" in plots:
        if results["auc_test"] is not False:
            plot_auc(logger, data, path, model)
            if os.path.isfile(path+'img/auc.png'):
                im = Image(path+'img/auc.png', 8 * inch, 3 * inch, kind='proportional')
                Report.append(im)
    
    if "gridsearch" in plots:
        if score is not False:
            plot_grid_search(logger, path)
            headers = list(score)
            columns = []
            for h in headers:
                if "param_" in h:
                    columns.append(h)
            for c in columns:
                if os.path.isfile(path+'img/grid_'+c+'.png'):
                    im = Image(path+'img/grid_'+c+'.png', 8 * inch, 3 * inch, kind='proportional')
                    Report.append(im)

    if "feature_importance" in plots or "feature_importance_full" in plots:
        value_test = 0
        if options.metric == 'mae':            
            value_test = results["mae_test"]
        elif options.metric == 'r2':
            value_test = results["r2_test"]
        elif options.metric == 'roc_auc':
            value_test = results["auc_test"]
        plot_features_importance(logger, options, data, model, path, value_test, options.metric)

    if "feature_importance" in plots:
        if os.path.isfile(path+'img/feature_importance.png'):
            im = Image(path+'img/feature_importance.png', 8 * inch, 8 * inch, kind='proportional')
            Report.append(im)

    if "feature_importance_full" in plots:
        if os.path.isfile(path+'img/feature_importance_full.png'):
            im = Image(path+'img/feature_importance_full.png', 8 * inch, 8 * inch, kind='proportional')
            Report.append(im)

    if "results" in plots and options.metric not in ['mae', 'r2']:
        plot_results(logger, options, data, pred_train, pred_test, pred_val, path)
        if os.path.isfile(path+'img/results.png'):
            a = Image(path+'img/results.png', 2 * inch, 2 * inch, kind='proportional')
            b = Image(path+'img/results_neg.png', 2 * inch, 2 * inch, kind='proportional')
            c = Image(path+'img/results_pos.png', 2 * inch, 2 * inch, kind='proportional')
            table_data = [[a, b, c]]
            results_table = Table(table_data, colWidths=2 * inch, rowHeights=2 * inch, normalizedData=1)
            Report.append(results_table)

    if "TSNE" in plots or "PCA" in plots:
        X = np.c_[data["x_train"].T, data["x_test"].T]
        X = np.c_[X, data["x_val"].T]
        X = X.T
        Y = np.c_[data["y_train"].T, data["y_test"].T]
        Y = np.c_[Y, data["y_val"].T]
        Y = Y.T

    if "TSNE" in plots:
        plot_rep_TSNE(logger, X, Y, path+'img/t-SNE.png')
        if os.path.isfile(path+'img/t-SNE.png'):
            im = Image(path+'img/t-SNE.png', 2 * inch, 2 * inch, kind='proportional')
            Report.append(im)
        
    if "PCA" in plots:
        plot_rep_PCA(logger, X, Y, path+'img/PCA.png')
        if os.path.isfile(path+'img/PCA.png'):
            im = Image(path+'img/PCA.png', 2 * inch, 2 * inch, kind='proportional')
            Report.append(im)
    
    Report.append(PageBreak())
    ptext = '<font size=24> <b> Dataset report </b> </font> <br/>'
    Report.append(Paragraph(ptext, styles["Justify"]))
    Report.append(Paragraph(" ", styles["Justify"]))
    Report.append(Paragraph(" ", styles["Justify"]))

    if "correlation" in plots:
        correlation(logger, X, path+'img/correlation.png')
        if os.path.isfile(path+'img/correlation.png'):
            im = Image(path+'img/correlation.png', 10 * inch, 10 * inch, kind='proportional')
            Report.append(im)
            Report.append(FrameBreak())
    
    if "distributions" in plots:
        distributions(logger, X, path+'img/distributions.png')
        if os.path.isfile(path+'img/distributions.png'):
            im = Image(path+'img/distributions.png', 8 * inch, 8 * inch, kind='proportional')
            Report.append(im)
    
    doc.build(Report)
    logger.info("Report complete, you can see it in the results folder")


if __name__ == "__main__":
    """Create dummy report."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    path = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp"
    accuracy_test = accuracy_train = accuracy_val = rec_train = 0
    rec_test = rec_val = auc_test = auc_train = auc_val = 0
    f1_train = f1_test = f1_val = rmse_test = rmse_train = rmse_val = 0
    mae_test = mae_train = mae_val = r2_test = r2_train = r2_val = 0
        
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

    train_proba, test_proba, val_proba = False, False, False
    model = False
    data = {
        'x_train': x_train,
        'x_test': x_test,
        'x_val': x_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val
        }

    results = {
        'accuracy_test': accuracy_test,
        'accuracy_train': accuracy_train,
        'accuracy_val': accuracy_val,
        'rec_train': rec_train,
        'rec_test': rec_test,
        'rec_val': rec_val,
        'auc_test': auc_test,
        'auc_train': auc_train,
        'auc_val': auc_val,
        'f1_train': f1_train,
        'f1_test': f1_test,
        'f1_val': f1_val,
        'rmse_test': rmse_test,
        'rmse_train': rmse_train,
        'rmse_val': rmse_val,
        'mae_test': mae_test,
        'mae_train': mae_train,
        'mae_val': mae_val,
        'r2_test': r2_test,
        'r2_train': r2_train,
        'r2_val': r2_val,
        'rparams': rparams
    }

    create_report(logger, path, train_proba, test_proba, val_proba, timer, rparams,
                  tstart, history, random_state, options, data, pred_train,
                  pred_test, pred_val, score, model, results)
