# Virtual screening
The project allows to use supervised learning on molecules written in the SMILES format by the training on SMILES themselves, maccs/morgan fingerprints and physical features using the following models:
- KNN
- Logistic regression
- Linear regression
- LSTM
- RandomForestClassifier
- SVC
- XGBClassifier

## Table of Contents
1. [Results](#results)
2. [Install](#install)
  1. [Conda](#conda)
  2. [Pip](#pip)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Input](#input)
6. [Output](#output)
7. [Citation](#citation)

## Results <a name="results"></a>

## Install <a name="install"></a>
- Linux
- Python 3.6+
- Parallel
  - apt-get install parallel or pacman -S parallel
- run env.sh
### with Conda <a name="conda"></a>
- Conda (https://www.anaconda.com/download/#linux)
- conda install --file requirements
- conda install -c conda-forge argh
- conda install -c rdkit rdkit
### with Pip <a name="pip"></a>
- Packages from requirements
- Argh (https://pypi.python.org/pypi/argh/)
- RDKit (https://github.com/rdkit/rdkit)


## Usage <a name="usage"></a>
    usage: logreg.py data section [-h] [--features FEATURES] [-o OUTPUT] [-c CONFIGS] [--fingerprint FINGERPRINT] [--n-bits N_BITS] [--n-jobs N_JOBS] [-p PATIENCE] [-g] [--dummy]

    positional arguments:
      data                  path to dataset
      section               name of section in config file

    optional arguments:
      -h, --help            show this help message and exit
      --features FEATURES   take features: all, fingerptint or physical (default: 'all')
      -o OUTPUT, --output OUTPUT
                        path to output directory (default: '/virtual_screening/tmp/')
      -c CONFIGS, --configs CONFIGS
                        path to config file (default: '/virtual_screening/src/configs/configs.ini')
      --fingerprint FINGERPRINT
                        maccs (167) or morgan (n) (default: 'morgan')
      --n-bits N_BITS       number of bits in Morgan fingerprint (default: 256)
      --n-jobs N_JOBS       number of jobs (default: 1)
      -p PATIENCE, --patience PATIENCE
                        patience of fit (default: 100)
      -g, --gridsearch      use gridsearch (default: False)
      --dummy               use only first 1000 rows of dataset (default: False)

## Dataset


## Example input <a name="input"></a>
python src/scripts/logreg.py data/tox21.csv LOGREG_TOX21 --features a --fingerprint morgan --n-bits 100 --n-jobs -1 -p 20 -t 1

## Example output <a name="output"></a>
Loading data <br />
Data shape: (7981, 308) <br />
Features shape: (7981, 296) <br />
Labels shape: (7981, 12) <br />
Data loaded <br />
X_train: (4126, 292) <br />
Y_train: (4126, 1) <br />
X_test: (1376, 292) <br />
Y_test: (1376, 1) <br />
X_val: (1376, 292) <br />
Y_val: (1376, 1) <br />
FIT <br />
EVALUATE <br />
Accuracy test: 91.28% <br />
Accuracy train: 95.27% <br />
0:06:27.941201 <br />
Can't create history plot for this type of experiment <br />
Report complete, you can see it in the results folder <br />
Done <br />
Results path /tmp/2017-12-01 17:45:27.798223 logreg tox21 0.913/ <br />


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
