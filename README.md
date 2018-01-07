# Virtual screening
The project allows to use supervised learning on molecules written in the SMILES format by the training on maccs/morgan fingerprints and physical features using the following models:
- KNN
- Logistic regression
- Linear regression
- RandomForestClassifier
- SVC
- XGBClassifier
- Isolation Forest

Use src/config.ini to configure the models.

## Table of Contents
1. [Results](#results)
2. [Install](#install)
    - [Conda](#conda)
    - [Pip](#pip)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Input](#input)
6. [Output](#output)
7. [Experiments table](#table)
8. [Citation](#citation)

## Results <a name="results"></a>
### Tox21 (https://tripod.nih.gov/tox21/challenge/data.jsp)
Set | Classs | AR	| AR-LBD	| AhR	| Aromatase	| ER	| ER-LBD	| PPAR-g | ARE	 | ATAD5	| HSE	 | MMP | p53
 --- | --- | --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Train | Inactive | 7197	|	6702	|	5948	|	5669	|	5631	|	6818	|	6422	|	5015	|	7003	|	6260	|	5018	|	6511
Train | Active | 270	|	224	|	767	|	296	|	702	|	319	|	184	|	943	|	252	|	356	|	922	|	419
Test | Inactive | 7106	| 6647	| 5879	| 5617	| 5504	| 6730	| 6377	| 4968	| 6944	| 6202	| 4971 |	6457
Test | Active | 306	| 231	| 785	| 305	| 793	| 350	| 186	| 958	| 265	| 378	| 927	| 425





## Install <a name="install"></a>
- Linux
- Python 3.6+ (Python 2.7 unstable)
- run env.sh
### with Conda <a name="conda"></a>
- Conda (https://www.anaconda.com/download/#linux)
- conda install --file requirements
- (optional) conda install -c akode xgboost
- conda install -c rdkit rdkit
### with Pip <a name="pip"></a>
- Packages from requirements
- (optional) xgboost
- RDKit (https://github.com/rdkit/rdkit)

## Usage <a name="usage"></a>
    usage: logreg.py data section [-h]
                          [--features {all,a,fingerprint,f,physical,p}]
                          [--output OUTPUT] [--configs CONFIGS]
                          [--fingerprint FINGERPRINT] [--n_bits N_BITS]
                          [--n_iter N_ITER] [--n_jobs N_JOBS]
                          [--patience PATIENCE] [--gridsearch GRIDSEARCH]
                          [--dummy DUMMY] [--targets TARGETS]
                          [--experiments_file EXPERIMENTS_FILE]
                          select_model [select_model ...] data [data ...]
                          section [section ...]

    positional arguments:
    select_model          name of the model, select from list in README
    data                  path to dataset
    section               name of section in config file

    optional arguments:
    -h, --help            show this help message and exit
    --features {all,a,fingerprint,f,physical,p}
                    take features: all, fingerptint or physical
    --output OUTPUT       path to output directory
    --configs CONFIGS     path to config file
    --fingerprint FINGERPRINT
                    maccs (167) or morgan (n)
    --n_bits N_BITS       number of bits in Morgan fingerprint
    --n_iter N_ITER       number of iterations in RandomizedSearchCV
    --n_jobs N_JOBS       number of jobs
    --patience PATIENCE, -p PATIENCE
                    patience of fit
    --gridsearch GRIDSEARCH, -g GRIDSEARCH
                    use gridsearch
    --dummy DUMMY, -d DUMMY
                    use only first 1000 rows of dataset
    --targets TARGETS, -t TARGETS
                    set number of target column
    --experiments_file EXPERIMENTS_FILE, -e EXPERIMENTS_FILE
                    where to write results of experiments

## Dataset <a name="dataset"></a>
A csv format file is required, in which one of the headers will be "smiles", and the rest - the names of the experiments(targets). The column "mol_id" will be dropped if exist. After processing, the names of the targets are saved, and instead of the "smiles", columns of fingerprints 'f' and physical representations 'p' are added.

## Example input <a name="input"></a>
python src/run_model.py logreg data/tox21.csv LOGREG_TOX21 --features all --fingerprint morgan --n_bits 1024 --n_jobs -1 -p 2000 -t 0 --n_iter 10

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
Results path /tmp/2017-12-01 17:45:27.798223 logreg tox21 all 0.913/ <br />

## Processing the experiment table  <a name="table"></a>
  1. Fill in the table with experiments parameters (examples in /etc, False = empty cell)
  2. Run run.py with Python, seriously
  3. Experiments will be performed one by one and fill in the columns with the results

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
