#!/usr/bin/env bash

conda install -y numpy
conda install -y pandas
conda install -y keras
conda install -y scikit-learn
conda install -y matplotlib
# conda install -y reportlab
conda install -y seaborn
conda install -y joblib
conda install -y -c conda-forge xgboost
conda install -y -c rdkit rdkit
conda install -y -c mordred-descriptor mordred
conda install -y cython
conda install -y theano
conda install -y -c conda-forge lightgbm
conda install -y -c glemaitre imbalanced-learn
conda install -y -c openbabel openbabel
conda install -y -c anaconda seaborn

yes | pip install argparse
yes | pip install configparser
yes | pip install pybel
yes | pip install pylatex
