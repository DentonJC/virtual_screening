#!/usr/bin/env bash

conda install -y numpy
conda install -y pandas
conda install -y keras
conda install -y scikit-learn
conda install -y matplotlib
conda install -y reportlab
conda install -y seaborn
conda install -y joblib
conda install -y -c conda-forge xgboost=0.6
conda install -y -c openbabel openbabel
conda install -y -c rdkit rdkit
conda install -y -c mordred-descriptor mordred
conda install -y cython
conda install -y theano

yes | pip install argparse
yes | pip install configparser
