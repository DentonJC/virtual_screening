# Development Log
### 01.12.17
- installation procedure (Conda)
  - https://github.com/DentonJC/virtual_screening/issues/6
- test on Ununtu
- bug in physical representation!
  - may be the main problem of physicals features results
- problems with paths and folders
  - https://github.com/DentonJC/virtual_screening/issues/4
  - https://github.com/DentonJC/virtual_screening/issues/3
- README.md updated

### 02.12.17
- small changes is outputs
- code cleaning

### 04.12.17
- README updated

### 06.12.17
- Table of Contents
- compare new physical features with previous - there <b>is</b> a difference! Better normalization required.
- create new datasets and start experiments on server

### 07.12.17
- total remove write_experiment
  - https://github.com/DentonJC/virtual_screening/issues/14
- rewrite run.py
  - https://github.com/DentonJC/virtual_screening/issues/13

### 08.12.17
- rewrite on argparse
- new arguments
- remove argh from dependences
- do not enter the values of the previous experiment into the table
  - https://github.com/DentonJC/virtual_screening/issues/17

### 09.12.17
- get rid of duplicate code in /scripts
  - https://github.com/DentonJC/virtual_screening/issues/9
- move the number of iterations of random search to the script arguments.
  - https://github.com/DentonJC/virtual_screening/issues/20
- run.py interact with run_model.py directly, no more tmp files!

### 10.12.17
- set nBits for maccs
  - https://github.com/DentonJC/virtual_screening/issues/18
- README updated
  - https://github.com/DentonJC/virtual_screening/issues/21
  - new run, usage
- \__main__ updated
- remove gridsearch.py

### 11.12.17
- xgboost library now optional
- Python2 compatibility
  - https://github.com/DentonJC/virtual_screening/issues/32

### 13.12.17
- error in start_time from run.py
  - two folders with same name if same start_time
- one gridsearch for all experiments in row
  - https://github.com/DentonJC/virtual_screening/issues/26

### 14.12.17
- turn off one gridsearch for all experiments in row
- train_test_split now with one random_state for all experiments in table
- better hyperparams in report (only rparams[0])

### 15.12.17
- two experiments columns for HIV dataset
- clean print + logging pairs
  - https://github.com/DentonJC/virtual_screening/issues/10
- try isolation forest
  - https://github.com/DentonJC/virtual_screening/issues/27
- add random forest to tables

### 16.12.17
- errors after testing with /etc/test.csv
  - AttributeError: 'RandomizedSearchCV' object has no attribute 'get'
- random_state added to report
- logging fixed

### 17.12.17
- Replace train accuracy with recall in the experiments tables.
  - https://github.com/DentonJC/virtual_screening/issues/34
- run.py now working only without argument
- Add choices to the argument options.
  - https://github.com/DentonJC/virtual_screening/issues/29
- remove xgb from tables - too long without interesting results
