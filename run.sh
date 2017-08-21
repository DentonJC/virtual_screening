python src/scripts/regression.py --input data/preprocessed/HIV_maccs.csv -c src/configs/config_regression_hiv.py --output tmp/ --feature maccs
python src/scripts/regression.py --input data/preprocessed/muv_maccs.csv -c src/configs/config_regression_muv.py --output tmp/ --feature maccs
python src/scripts/regression.py --input data/preprocessed/tox21_maccs.csv -c src/configs/config_regression_tox21.py --output tmp/ --feature maccs
python src/scripts/residual.py --input data/preprocessed/HIV_maccs.csv -c src/configs/config_residual_hiv.py --output tmp/ --feature maccs
python src/scripts/residual.py --input data/preprocessed/muv_maccs.csv -c src/configs/config_residual_muv.py --output tmp/ --feature maccs
python src/scripts/residual.py --input data/preprocessed/tox21_maccs.csv -c src/configs/config_residual_tox21.py --output tmp/ --feature maccs