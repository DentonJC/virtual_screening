import os
import csv
import numpy as np


def main(FILENAME):
    # Mod, bits, decr
    grouped = {}
    with open(FILENAME+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
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
                val = float(row[i]) if i > 2 else row[i]
                grouped[k][header[i]].append(val)

    with open(FILENAME+"_all_results.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        writer.writerow(header)
        for k in grouped:
            values = [grouped[k]['Model'][0], grouped[k]['Descriptors'][0], grouped[k]['n_bits'][0]]

            for i in range(3, len(header)):
                mean = np.mean(grouped[k][header[i]])
                error = abs(min(grouped[k][header[i]]) - max(grouped[k][header[i]]))/2
                values = values + ["%.5f±%.5f" % (mean, error)]
            writer.writerow(values)

    # Mod
    grouped = {}
    with open(FILENAME+'.csv', newline='') as csvfile:
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
                val = float(row[i]) if i > 2 else row[i]
                grouped[k][header[i]].append(val)

    with open(FILENAME+"_model_results.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        writer.writerow(['Model'] + header[3:])
        for k in grouped:
            values = [grouped[k]['Model'][0]]

            for i in range(3, len(header)):
                mean = np.mean(grouped[k][header[i]])
                error = abs(min(grouped[k][header[i]]) - max(grouped[k][header[i]]))/2
                values = values + ["%.5f±%.5f" % (mean, error)]
            writer.writerow(values)

    # bits, decr
    grouped = {}
    with open(FILENAME+'.csv', newline='') as csvfile:
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
                val = float(row[i]) if i > 2 else row[i]
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


if __name__ == "__main__":
    root_address = os.path.dirname(os.path.realpath(__file__)).replace("/utils", "")
    NAME = ["clintox_scaffold", "clintox_random", "clintox_cluster", "bace_scaffold", "bace_random", "bace_cluster"]
    for n in NAME:
        print(n)
        main(root_address+'/etc/'+n)
