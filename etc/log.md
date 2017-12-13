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

### 12.12.17
