# Tables of experiments

- The names and number of input columns should match the list of parameters in run.py (and moloi.py)
- The number of result columns should match the length and order of the list of outputs in run.py
- The config addresses start at the root of the program and lead to the corresponding folders
- The correct names for models, splits and descriptors are specified in the README of the project
- The purpose of the columns and the acceptable value can be printed using: python moloi.py --help
- An empty field in Gridsearch means False
