python src/scripts/regression.py --input data/preprocessed/muv_maccs.csv -c src/configs/config_regression_muv.py --output tmp/ --feature maccs -g
python src/scripts/residual.py --input data/preprocessed/muv_morgan.csv -c src/configs/config_residual_muv.py --output tmp/ --feature morgan -g
python src/scripts/regression.py --input data/preprocessed/muv_morgan.csv -c src/configs/config_regression_muv.py --output tmp/ --feature morgan -g
python src/scripts/residual.py --input data/preprocessed/muv_maccs.csv -c src/configs/config_residual_muv.py --output tmp/ --feature maccs -g
