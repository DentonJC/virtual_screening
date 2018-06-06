# Virtual screening
## About

Descriptors:
- MACCS (maccs)
- morgan/ECFP (morgan)
- RDKit (rdkit)
- mordred (mordred)
- spectrophore (spectrophore)

Models:
- KNN (knn)
- Logistic regression (lr)
- RandomForestClassifier (rf)
- SVC (svc)
- XGBClassifier (xgb)
- Isolation Forest (if)
- FCNN (fcnn)

Splits:
- random
- stratified
- scaffold
- cluster

## Goals
- Seq2Seq -> transfer learning
- exclude the rich get richer
- feature selection

## Table of Contents
1. [Results](#results)
2. [Install](#install)
    - [Conda](#conda)
    - [Pip](#pip)
3. [Usage](#usage)
4. [Input](#input)
5. [Output](#output)
6. [Datasets](../master/data/README.md)
7. [Data config](../master/data/data_configs/README.md)
8. [Model config](../master/data/model_configs/README.md)
9. [Single experiment](../master/moloi/bin/README.md)
10. [Experiments table](../master/etc/README.md)
10. [Utilities](../master/utils/README.md)
11. [Citation](#citation)

## Results <a name="results"></a>
[BACE](../master/etc/preprocessed/experiments_bace_results.md)

## Install <a name="install"></a>
- Linux
- Python 3.6+ (Python 2.7 unstable)
- source env.sh
- It is better to use Theano backend.
### with Conda <a name="conda"></a>
- sh setup.sh

  or
- Conda (https://www.anaconda.com/download/#linux)
- conda install --file requirements
- conda install -c conda-forge xgboost
- conda install -c openbabel openbabel
- conda install -c rdkit rdkit
- conda install -c mordred-descriptor mordred
- Python3: pip install configparser
- Python2: pip install ConfigParser
- pip install argparse
### with Pip <a name="pip"></a>
- pip install git+git://github.com/DentonJC/virtual_screening

  or
- Packages from requirements
- pip install xgboost
- RDKit (https://github.com/rdkit/rdkit)
- pip install mordred
- Python3: pip install configparser
- Python2: pip install ConfigParser
- pip install argparse

## Usage <a name="usage"></a>
    usage: model data section [-h] [--select_model SELECT_MODEL]
                      [--data_config DATA_CONFIG] [--section SECTION]
                      [--load_model LOAD_MODEL]
                      [--descriptors DESCRIPTORS] [--output OUTPUT]
                      [--model_config MODEL_CONFIG] [--n_bits N_BITS]
                      [--n_cv N_CV] [--n_iter N_ITER] [--n_jobs N_JOBS]
                      [--patience PATIENCE] [--gridsearch]
                      [--metric {accuracy,roc_auc,f1,matthews}]
                      [--split_type {stratified,scaffold,random,cluster}]
                      [--split_size SPLIT_SIZE] [--targets TARGETS]
                      [--experiments_file EXPERIMENTS_FILE]

    optional arguments:
    -h, --help            show this help message and exit
    --select_model SELECT_MODEL
                    name of the model, select from list in README
    --data_config DATA_CONFIG
                    path to dataset config file
    --section SECTION     name of section in model config file
    --load_model LOAD_MODEL
                    path to model .sav
    --descriptors DESCRIPTORS
                    descriptor of molecules
    --output OUTPUT       path to output directory
    --model_config MODEL_CONFIG
                    path to config file
    --n_bits N_BITS       number of bits in Morgan fingerprint
    --n_cv N_CV           number of splits in RandomizedSearchCV
    --n_iter N_ITER       number of iterations in RandomizedSearchCV
    --n_jobs N_JOBS       number of jobs
    --patience PATIENCE, -p PATIENCE
                    patience of fit
    --gridsearch, -g      use gridsearch
    --metric {accuracy,roc_auc,f1,matthews}
                    metric for RandomizedSearchCV
    --split_type {stratified,scaffold,random,cluster}
                    type of train-test split
    --split_size SPLIT_SIZE     size of test and valid splits
    --targets TARGETS, -t TARGETS
                    set number of target column
    --experiments_file EXPERIMENTS_FILE, -e EXPERIMENTS_FILE
                    where to write results of experiments

## Processing the experiment table  <a name="table"></a>
  1. Fill in the table with experiments parameters (examples in /etc, False = empty cell), UTF-8
  2. Run run.py with Python, seriously
  3. Experiments will be performed one by one and fill in the columns with the results

## Single experiment

## Example input <a name="input"></a>
    python moloi/moloi.py --model_config '/data/model_configs/configs.ini' --descriptors ['rdkit', 'morgan','mordred', 'maccs'] --n_bits 2048 --n_cv 5 -p 100 -g --n_iter 300 --metric 'roc_auc' --split_type 'scaffold' --split_s 0.1 --select_model 'rf' --data_config '/data/data_configs/bace.ini' --section 'RF' -e 'etc/experiments_bace.csv' -t 0

## Example output <a name="output"></a>

Script adderss: run.py <br />
Descriptors: ['rdkit', 'morgan', 'mordred', 'maccs'] <br />
n_bits: 2048 <br />
Config file: /data/model_configs/configs.ini <br />
Section: RF <br />
Grid search <br />
Load train data <br />
Load test data <br />
Load val data <br />
Data loaded <br />
x_train shape: (1207, 4239) <br />
x_test shape: (152, 4239) <br />
x_val shape: (154, 4239) <br />
y_train shape: (1207, 1) <br />
y_test shape: (152, 1) <br />
y_val shape: (154, 1) <br />
GRID SEARCH <br />
GRIDSEARCH FIT <br />
MODEL FIT <br />
EVALUATE <br />
Accuracy test: 70.39% <br />
0:07:37.644208 <br />
Creating report <br />
Report complete, you can see it in the results folder <br />
Results path: /tmp/2018-05-27_15:45:04_RF_['rdkit','morgan','mordred','maccs']70.395/ <br />
Done <br />

## Citation <a name="citation"></a>
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
- Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array: A Structure for Efficient Numerical Computation, Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37 (publisher link)
- Travis E. Oliphant. Python for Scientific Computing, Computing in Science & Engineering, 9, 10-20 (2007), DOI:10.1109/MCSE.2007.58 (publisher link)
- K. Jarrod Millman and Michael Aivazis. Python for Scientists and Engineers, Computing in Science & Engineering, 13, 9-12 (2011), DOI:10.1109/MCSE.2011.36 (publisher link)
- Fernando Pérez and Brian E. Granger. IPython: A System for Interactive Scientific Computing, Computing in Science & Engineering, 9, 21-29 (2007), DOI:10.1109/MCSE.2007.53 (publisher link)
- John D. Hunter. Matplotlib: A 2D Graphics Environment, Computing in Science & Engineering, 9, 90-95 (2007), DOI:10.1109/MCSE.2007.55 (publisher link)
- Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010) (publisher link)
- O. Tange (2011): GNU Parallel - The Command-Line Power Tool,  ;login: The USENIX Magazine, February 2011:42-47.
- https://github.com/gmum/ananas/blob/master/fingerprints/_desc_rdkit.py
- RDKit: Open-source cheminformatics; http://www.rdkit.org
- Keras (2015), Chollet et al., https://github.com/fchollet/keras
