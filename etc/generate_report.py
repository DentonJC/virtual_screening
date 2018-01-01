#!/usr/bin/python3

import re
import sys


def sortby(entries, parameter):
    '''
    Returns entries list sorted by parameter
    Example: entries1 = sortby(entries, 'NR-AR-LBD')
    '''
    return sorted(entries, key=lambda entry: -entry[parameter])


def main(exp_address, report_address, n_params):
    data=[]
    for line in open(exp_address, 'r'):
        data.append(line.strip().split(','))
    entries = [dict(zip(data[0], data[i + 1])) for i in range(len(data) - 1)]
    tmp_p = data[0][n_params:]
    params = []
    params_formatted = []

    for i in range((len(tmp_p)+1)//2):
        params.append([tmp_p[2*i], tmp_p[2*i + 1]])
        params_formatted.append(re.search('^([\w-]+)', tmp_p[2*i]).group(1))

    for entry in entries:
        for p in range(len(params_formatted)):
            sum_val = float(entry[params[p][0]]) + float(entry[params[p][1]])
            entry[params_formatted[p]]=sum_val


    with open(report_address, 'w') as ratings_file:
        for p in params_formatted:
            print('\t\t' + p, file=ratings_file)
            for entry in sortby(entries, p):
                print(('{Model}\t{' + p + '}').format(**entry), file=ratings_file)
            print('', file=ratings_file)
            
if __name__ == "__main__":
    exp_address = 'experiments_tox21.csv'
    report_address = 'report.txt'
    n_params = 13
    main(exp_address, report_address, n_params)
