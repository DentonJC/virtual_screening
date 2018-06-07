#!/usr/bin/env python

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from pylab import xlim
plt.style.use('ggplot')


NAME = "bace"
FILENAME = 'experiments_'+NAME
split = "scaffold"
split_s = 0.1
lims = [0.5, 1]
root_address = os.path.dirname(os.path.realpath(__file__)).replace("/utils", "")


grouped = {}
cfile = pd.read_csv(root_address+"/etc/"+FILENAME+".csv")
cfile.to_csv(root_address+"/etc/"+FILENAME+"_bkp.csv", sep=",", index=False)
try:
    cfile = cfile[pd.notnull(cfile['Load model'])]
except:
    pass
header = list(cfile)
for i in ['gparams','Data config', 'Model config', 'Section', 'Output', 'n_jobs', 'Load model', 'Metric', 'Split type', 'Split size', 'n_cv', 'Patience', 'Gridsearch', 'n_iter']:
        #print(i, "removed")
        try:
            header.remove(i)
            cfile = cfile.drop(i, 1)
        except:
            pass

cfile.to_csv(root_address+"/etc/"+FILENAME+".csv", sep=",", index=False)
header = list(cfile)
header.remove('Descriptors')

with open(root_address+"/etc/"+FILENAME+".csv", newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    
    for row in spamreader:
        if row[0] == 'Model': continue
        k = row[1]
        if k not in grouped: grouped[k] = {'vals': []}
        if 'morgan' not in k and 'spectrophore' not in k:
            grouped[k]['vals'].append(row[:1]+[" "]+row[3:])
        else:
            grouped[k]['vals'].append(row[:1]+row[2:])

for desc in grouped:
    directory = FILENAME+'_'+desc
    if not os.path.exists(root_address+"/etc/preprocessed/"+directory):
        os.makedirs(root_address+"/etc/preprocessed/"+directory)

    for j in range(2,len(header)):
        #print(header[j])
        grouped2 = dict()
        for row in grouped[desc]['vals']:
            #print(row, "J:", j)
            if not row[j] or row[j] == '': continue

            k = row[0]+" "+row[1]
            #print(k)
            if k not in grouped2: grouped2[k] = {'vals': []}
            grouped2[k]['vals'].append(float(row[j]))
            grouped2[k]['Model'] = row[0]
            if 'morgan' not in desc and 'spectrophore' not in desc:
                grouped2[k]['n_bits'] = " "
            else:
                grouped2[k]['n_bits'] = row[1]
			
        csv_file = open(root_address+"/etc/preprocessed/"+directory+"/%s_grouped.csv" % header[j], "w", newline='')
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        writer.writerow(["Model", "n_bits", header[j], "error"])
        for k in grouped2:
            writer.writerow([grouped2[k]['Model'], grouped2[k]['n_bits'], 
                np.mean(grouped2[k]['vals']), abs(min(grouped2[k]['vals']) - max(grouped2[k]['vals']))/2])
        csv_file.close()


def label_barh(root_address, fullname, path, FILENAME, split, split_s, text_format, is_inside=True, lims=[0, 1], filename="1", title='RDKit+Mordred', descript=''):
    results = pd.read_csv(path+'/'+filename)
    cols = list(results)
    cols.remove("Model")
    try:
        cols.remove("Descriptors")
    except:
        pass
    cols.remove("n_bits")
    print(cols)
    #for i in range(0, len(cols), 2):
    #    col = cols[i]
    #    err_col = cols[i+1]
    #    print(col)
    fig, ax = plt.subplots()
    data = results["Model"].map(str) + " " + results["n_bits"].map(str)

    #    performance = results[col]
    #    yerr = results[err_col]
    performance = results[cols[0]]
    xerr = results[cols[1]]
    
    y_pos = np.arange(len(data))
    bar = ax.barh(y_pos, performance, align='center', xerr=xerr, error_kw=dict(elinewidth=2, color='g'))

    colors = ["C0", "C1", "C8", "C3", "C4", "C2", "C7"]

    for i, b in enumerate(bar):
        if 'knn' in data[i]:
            b.set_color(colors[0])
        if 'regression' in data[i]:
            b.set_color(colors[1])
        if 'lr' in data[i]:
            b.set_color(colors[2])
        if 'svc' in data[i]:
            b.set_color(colors[3])
        if 'xgb' in data[i]:
            b.set_color(colors[4])
        if 'rf' in data[i]:
            b.set_color(colors[5])
        if 'fcnn'in data[i]:
            b.set_color(colors[6])
                
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data)
    ax.invert_yaxis()  
    ax.set_xlabel(cols[0])
    ax.set_title(fullname+title)
    max_y_value = max(b.get_height() for b in bar)
    distance = max_y_value * 0.05

    for b in bar:
        text = text_format.format(b.get_width())
        """
        if is_inside:
            text_x = b.get_width() - distance
        else:
            text_x = b.get_width() + distance
        """
        text_x = lims[0] + lims[0] * 0.05
        text_y = b.get_y() + b.get_height() / 2

        ax.text(text_x, text_y, text, va='center', color='#FFFFFF', size=7)   
    plt.xlim(lims)
    #print()
    plt.savefig(root_address+"/etc/preprocessed/experiments_"+FILENAME+'_'+split+'_'+str(split_s)+'_'+descript.replace('[','').replace(']','').replace('\'','')+'_'+cols[0]+'.png', dpi=600)
    plt.close()
    #plt.show()

filenames=["maccs", "rdkit", "mordred", "rdkit_maccs", "rdkit_mordred", "morgan_maccs", "morgan_mordred", "rdkit_morgan", "morgan", "rdkit_morgan_mordred_maccs", "mordred_maccs"]
titles=['MACCS', 'RDKit', 'Mordred', 'MACCS+RDKit', 'RDKit+Mordred', 'Morgan+MACCS', 'Morgan+Mordred', 'RDKit+Morgan', 'Morgan', 'RDKit+Morgan+Mordred+MACCS', 'Mordred+MACCS']

for i in range(len(titles)):
    try:
        print(titles[i])
        try:
            path = root_address+"/etc/preprocessed/experiments_"+NAME+"_['"+filenames[i].replace("_","','")+"']"
            onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        except:
            path = root_address+"/etc/preprocessed/experiments_"+NAME+"_['"+filenames[i].replace("_","', '")+"']"
            onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for f in onlyfiles:
            fullname = NAME + ', ' + split + " ("+str(1-2*split_s)+", "+str(split_s)+", "+str(split_s)+"), "
            label_barh(root_address, fullname, path, NAME, split, split_s, text_format="{:4.5f}", is_inside=True, lims=lims, filename=f, title=titles[i], descript="['"+filenames[i].replace("_","','")+"']")
    except:
        print(titles[i]+" failed")
