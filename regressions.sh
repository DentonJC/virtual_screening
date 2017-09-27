python src/scripts/regression.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g

python src/scripts/regression_etc.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g

python src/scripts/regression_pca_10.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/regression_pca_50.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/regression_pca_100.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/regression_pca_500.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g

python src/scripts/regression_rfe_10.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/regression_rfe_50.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/regression_rfe_100.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/regression_rfe_500.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g

python src/scripts/regression_ust_10.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/regression_ust_50.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/regression_ust_100.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/regression_ust_500.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1000 -g




python src/scripts/regression.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g

python src/scripts/regression_etc.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g

python src/scripts/regression_pca_10.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g
python src/scripts/regression_pca_50.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g
python src/scripts/regression_pca_100.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g

python src/scripts/regression_rfe_10.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g
python src/scripts/regression_rfe_50.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g
python src/scripts/regression_rfe_100.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g

python src/scripts/regression_ust_10.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g
python src/scripts/regression_ust_50.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g
python src/scripts/regression_ust_100.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs -g