# Virtual screening
It is better to use Theano backend.
Descriptors:
- MACCS
- morgan (ECFP)
- RDKit
- mordred
- spectrophore

Models:
- KNN
- Logistic regression
- RandomForestClassifier
- SVC
- XGBClassifier
- Isolation Forest
- FCNN

Splits:
- random
- stratified
- scaffold
- cluster

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
### Tox21
link: https://tripod.nih.gov/tox21/challenge/data.jsp

Set | Class | AR	| AR-LBD	| AhR	| Aromatase	| ER	| ER-LBD	| PPAR-g | ARE	 | ATAD5	| HSE	 | MMP | p53
 --- | --- | --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Train | Inactive | 7197	|	6702	|	5948	|	5669	|	5631	|	6818	|	6422	|	5015	|	7003	|	6260	|	5018	|	6511
Train | Active | 270	|	224	|	767	|	296	|	702	|	319	|	184	|	943	|	252	|	356	|	922	|	419
Test | Inactive | 574 | 574 | 537 | 489 | 465 | 580 | 574 | 462 | 584 | 588 | 483 | 575 |
Test | Active | 12  | 8   | 73  | 39  | 51  | 20  | 31  | 93  | 38  | 22  | 60  | 41  |

#### After RDKit featurization

Set | Class | AR	| AR-LBD	| AhR	| Aromatase	| ER	| ER-LBD	| PPAR-g | ARE	 | ATAD5	| HSE	 | MMP | p53
 --- | --- | --- |  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Train | Inactive | 6925	|	6441	|	5661	|	5438	|	5392	|	6522	|	6148	|	4849	|	6703	|	6048	|	4782	|	6239
Train | Active | 220	|	185	|	761	|	278	|	664	|	309	|	180	|	914	|	250	|	350	|	879	|	389
Test | Inactive | 565 | 566 | 529 | 482 | 458 | 570 | 564 | 454 | 574 | 577 | 474 | 566 |
Test | Active | 11  | 7   | 70  | 38  | 50  | 20  | 30  | 91  | 36  | 21  | 57  | 39  |


### MUV
link: https://jcheminf.springeropen.com/articles/10.1186/1758-2946-5-26

Class | MUV-466 | MUV-548 | MUV-600 | MUV-644 | MUV-652 | MUV-689 | MUV-692 | MUV-712 | MUV-713 | MUV-733 | MUV-737 | MUV-810 | MUV-832 | MUV-846 | MUV-852 | MUV-858 | MUV-859
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Inactive | 14999 | 15000 | 14999 | 14997 | 15000 | 14993 | 15000 | 14994 | 14990 | 14993 | 14996 | 14998 | 15000 | 14988 | 14991 | 15000 | 14999 |
Active | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30

#### After RDKit featurization
Class | MUV-466 | MUV-548 | MUV-600 | MUV-644 | MUV-652 | MUV-689 | MUV-692 | MUV-712 | MUV-713 | MUV-733 | MUV-737 | MUV-810 | MUV-832 | MUV-846 | MUV-852 | MUV-858 | MUV-859
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Inactive | 14999 | 15000 | 14999 | 14997 | 15000 | 14993 | 15000 | 14994 | 14990 | 14993 | 14996 | 14998 | 15000 | 14988 | 14991 | 15000 | 14999 |
Active | 30    | 30    | 30    | 30    | 30    | 30    | 30    | 30    | 30    | 30    | 30    | 30    | 30    | 30    | 30    | 30    | 30    |



### HIV
link: http://moleculenet.ai/datasets-1/

Class | Activity
--- | ---
Inactive | 39684
Active | 1443

#### After RDKit featurization
Class | Activity
--- | ---
Inactive | 39604
Active | 1441

### BACE
link: http://moleculenet.ai/datasets-1/

Set | Class | Activity
--- | --- | ---
Train | Inactive | 103
Train | Active | 145
Test | Inactive | 719
Test | Active | 546

#### After RDKit featurization
Set | Class | Activity
--- | --- | ---
Train | Inactive | 103
Train | Active | 145
Test | Inactive | 719
Test | Active | 546

## Install <a name="install"></a>
- Linux
- Python 3.6+ (Python 2.7 unstable)
- source env.sh
- pip install git+git://github.com/DentonJC/virtual_screening
### with Conda <a name="conda"></a>
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
- Packages from requirements
- pip install xgboost
- RDKit (https://github.com/rdkit/rdkit)
- pip install mordred
- Python3: pip install configparser
- Python2: pip install ConfigParser
- pip install argparse

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
<b>Attention:</b> Please, add '\_train', '\_test' and '\_val' to the end of files if you want to run the experiment table. This is not a bug, it is a feature.

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

## Processing the experiment table  <a name="table"></a>
  1. Fill in the table with experiments parameters (examples in /etc, False = empty cell), UTF-8
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
