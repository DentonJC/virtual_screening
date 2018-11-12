#!/usr/bin/env python

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from pylab import xlim
from textwrap import wrap
plt.style.use('ggplot')


def create_tables(FILENAME, INPUT):
    # Mod, bits, decr
    grouped = {}
    with open(INPUT+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        header = []
        for row in spamreader:
            if row[0] == 'Model':
                header = row
                continue
            k = row[0]+row[1]+row[2]
            if k not in grouped:
                grouped[k] = {}
                for head in header:
                    grouped[k][head] = []
            for i in range(len(header)):
                try:
                    val = float(row[i]) if i > 2 else row[i]
                except ValueError:
                    continue
                grouped[k][header[i]].append(val)

    with open(FILENAME+"_all_results.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        writer.writerow(header)
        for k in grouped:
            values = [grouped[k]['Model'][0], grouped[k]['Descriptors'][0], grouped[k]['n_bits'][0]]

            for i in range(3, len(header)):
                mean = np.mean(grouped[k][header[i]])
                try:
                    error = abs(min(grouped[k][header[i]]) - max(grouped[k][header[i]]))/2
                except ValueError:
                    continue
                values = values + ["%.5f±%.5f" % (mean, error)]
            writer.writerow(values)

    # Mod
    grouped = {}
    with open(INPUT+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if row[0] == 'Model':
                header = row
                continue
            k = row[0]
            if k not in grouped:
                grouped[k] = {}
                for head in header:
                    grouped[k][head] = []
            for i in range(len(header)):
                try:
                    val = float(row[i]) if i > 2 else row[i]
                except ValueError:
                    continue
                grouped[k][header[i]].append(val)

    with open(FILENAME+"_model_results.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        writer.writerow(['Model'] + header[3:])
        for k in grouped:
            values = [grouped[k]['Model'][0]]

            for i in range(3, len(header)):
                mean = np.mean(grouped[k][header[i]])
                try:
                    error = abs(min(grouped[k][header[i]]) - max(grouped[k][header[i]]))/2
                except ValueError:
                    continue
                values = values + ["%.5f±%.5f" % (mean, error)]
            writer.writerow(values)

    # bits, decr
    grouped = {}
    with open(INPUT+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if row[0] == 'Model':
                header = row
                continue
            k = row[2]+row[1]
            if k not in grouped:
                grouped[k] = {}
                for head in header:
                    grouped[k][head] = []
            for i in range(len(header)):
                try:
                    val = float(row[i]) if i > 2 else row[i]
                except ValueError:
                    continue
                grouped[k][header[i]].append(val)

    with open(FILENAME+"_descriptors_results.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        writer.writerow(header[1:])
        for k in grouped:
            values = [grouped[k]['Descriptors'][0], grouped[k]['n_bits'][0]]

            for i in range(3, len(header)):
                mean = np.mean(grouped[k][header[i]])
                error = abs(min(grouped[k][header[i]]) - max(grouped[k][header[i]]))/2
                values = values + ["%.5f±%.5f" % (mean, error)]
            writer.writerow(values)


def prep(NAME, FILENAME, root_address):
    grouped = {}
    cfile = pd.read_csv(root_address+'/etc/'+FILENAME+".csv")
    try:
        cfile = cfile[pd.notnull(cfile['Load model'])]
    except:
        pass
    header = list(cfile)
    for i in ['gparams', 'Data config', 'Model config', 'Section', 'Output',
              'n_jobs', 'Load model', 'Metric', 'Split type', 'Split size',
              'n_cv', 'Patience', 'Gridsearch', 'n_iter']:
            try:
                header.remove(i)
                cfile = cfile.drop(i, 1)
            except:
                pass

    cfile.to_csv(root_address+'/tmp/results/'+FILENAME+".csv", sep=",", index=False)
    header = list(cfile)
    header.remove('Descriptors')

    with open(root_address+'/tmp/results/'+FILENAME+".csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

        for row in spamreader:
            if row[0] == 'Model':
                continue
            k = row[1]
            if k not in grouped:
                grouped[k] = {'vals': []}
            if 'morgan' not in k and 'spectrophore' not in k:
                grouped[k]['vals'].append(row[:1]+[" "]+row[3:])
            else:
                grouped[k]['vals'].append(row[:1]+row[2:])

    for desc in grouped:
        directory = FILENAME+'_'+desc
        if not os.path.exists(root_address+"/tmp/results/preprocessed_"+NAME+"/"+directory):
            os.makedirs(root_address+"/tmp/results/preprocessed_"+NAME+"/"+directory)

        for j in range(2, len(header)):
            grouped2 = dict()
            for row in grouped[desc]['vals']:
                if not row[j] or row[j] == '':
                    continue
                k = row[0]+" "+row[1]
                if k not in grouped2:
                    grouped2[k] = {'vals': []}
                grouped2[k]['vals'].append(float(row[j]))
                grouped2[k]['Model'] = row[0]
                if 'morgan' not in desc and 'spectrophore' not in desc:
                    grouped2[k]['n_bits'] = " "
                else:
                    grouped2[k]['n_bits'] = row[1]

            csv_file = open(root_address + "/tmp/results/preprocessed_" + NAME + "/" +
                            directory + "/%s_grouped.csv" % header[j], "w", newline='')
            writer = csv.writer(csv_file, delimiter=',', quotechar='"')
            writer.writerow(["Model", "n_bits", header[j], "error"])
            for k in grouped2:
                writer.writerow([grouped2[k]['Model'], grouped2[k]['n_bits'],
                                np.mean(grouped2[k]['vals']),
                                abs(min(grouped2[k]['vals']) - max(grouped2[k]['vals']))/2])
            csv_file.close()


def label_barh(root_address, fullname, path, NAME, FILENAME, split, split_s, text_format, is_inside=True, lims=[0, 1], filename="1", title='RDKit+Mordred', descript=''):
    results = pd.read_csv(path+'/'+filename)
    cols = list(results)
    cols.remove("Model")
    try:
        cols.remove("Descriptors")
    except:
        pass
    cols.remove("n_bits")
    print(cols)
    # for i in range(0, len(cols), 2):
    #     col = cols[i]
    #     err_col = cols[i+1]
    #     print(col)
    fig, ax = plt.subplots()
    data = results["Model"].map(str) + " " + results["n_bits"].map(str)

    #    performance = results[col]
    #    yerr = results[err_col]
    performance = results[cols[0]]
    xerr = results[cols[1]]

    y_pos = np.arange(len(data))
    bar = ax.barh(y_pos, performance, align='center', xerr=xerr, alpha=0.5, error_kw=dict(capthick=1, lw=1, capsize=3, ecolor='#000083'))

    #colors = ["C0", "C1", "C8", "C3", "C4", "C2", "C7"]
    colors = ["red", "yellow", "blue", "orange", "magenta", "green", "gray"]

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
    data = [ '\n'.join(wrap(d, 7)) for d in data ]
    ax.set_yticklabels(data)
    ax.invert_yaxis()
    ax.set_xlabel(cols[0])
    ax.set_title("\n".join(wrap(fullname.split('_')[0]+title, 60)))
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

    plt.savefig(root_address + "/tmp/results/preprocessed_" + NAME + "/" + FILENAME +
                '_' + split + '_' + str(split_s) + '_' +
                descript.replace('[', '').replace(']', '').replace('\'', '') +
                '_' + cols[0]+'.png', dpi=600)
    plt.close()
    # plt.show()
    return '_' + split + '_' + str(split_s) + '_' +descript.replace('[', '').replace(']', '').replace('\'', '') + '_' + cols[0] + '.png'


def create_result_plots(filenames, titles, NAME, FILENAME, split_s, split, lims, root_address):
    prep(NAME, FILENAME, root_address)
    with open(root_address+"/tmp/results/preprocessed_"+NAME+"/"+FILENAME+'_results.md', 'w') as results:
        results.write('## '+FILENAME+'\n')
    for i in range(len(titles)):
        if True:
            print()
            print(titles[i])
            try:
                # path = root_address + "/tmp/results/preprocessed_" + NAME + "/" + NAME + "_['" + filenames[i].replace("_", "','") + "']"
                path = root_address + "/tmp/results/preprocessed_" + NAME + "/" + NAME + '_' + filenames[i].replace("_", ", ")
                onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
                for f in onlyfiles:
                    fullname = NAME + ', ' + split + " ("+str(1-2*split_s)+", "+str(split_s)+", "+str(split_s)+"), "
                    addr = label_barh(root_address, fullname, path, NAME, FILENAME, split, split_s, text_format="{:4.5f}", is_inside=True, lims=lims, filename=f, title=titles[i], descript="['" + filenames[i].replace("_", "','") + "']")
                    if 'val' in addr:
                        with open(root_address+"/tmp/results/preprocessed_"+NAME+"/"+FILENAME+'_results_val.md', 'a') as results:
                            results.write('<img src="../preprocessed_'+NAME+"/"+FILENAME+addr+'" /><br/>\n')
                    else:
                        with open(root_address+"/tmp/results/preprocessed_"+NAME+"/"+FILENAME+'_results.md', 'a') as results:
                            results.write('<img src="../preprocessed_'+NAME+"/"+FILENAME+addr+'" /><br/>\n')
            except Exception as e:
                print("Can not plot " + titles[i] + ": " + str(e))

def process_results(filenames, titles, NAME):
    root_address = os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "")
    if not os.path.exists(root_address+'/tmp/results/'):
        os.makedirs(root_address+'/tmp/results/')

    split_s = 0.1
    lims = [0.5, 1]
    
    for n in NAME:
        FILENAME = n
        if 'scaffold' in n:
            s = 'scaffold'
        elif 'cluster' in n:
            s = 'cluster'
        elif 'random' in n:
            s = 'random'
        else:
            s = ''
        print(n, s)
        create_result_plots(filenames, titles, n, FILENAME, split_s, s, lims, root_address)
        
    for n in NAME:
        print(n)
        create_tables(root_address+'/tmp/results/'+n, root_address+'/tmp/results/'+n)


if __name__ == "__main__":
    filenames = ["maccs", "rdkit", "mordred", "morgan", "rdkit_maccs", "rdkit_mordred", "morgan_maccs", "morgan_mordred", "rdkit_morgan", "mordred_maccs", "rdkit_morgan_mordred_maccs"]
    titles = ['MACCS', 'RDKit', 'Mordred', 'Morgan', 'MACCS+RDKit', 'RDKit+Mordred', 'Morgan+MACCS', 'Morgan+Mordred', 'RDKit+Morgan', 'Mordred+MACCS', 'RDKit+Morgan+Mordred+MACCS']
    # NAME = ["clintox_scaffold", "clintox_random", "clintox_cluster", "bace_scaffold", "bace_random", "bace_cluster"]
    NAME = ["test", "clintox"]
    process_results(filenames, titles, NAME)
