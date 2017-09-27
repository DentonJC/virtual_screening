python src/scripts/residual.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/residual.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature maccs -g

python src/scripts/residual_etc.py --input data/preprocessed/tox21_morgan_1000.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature morgan --n 1000 -g
python src/scripts/residual_etc.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature maccs -g