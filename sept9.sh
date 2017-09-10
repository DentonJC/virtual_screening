python src/scripts/regression.py --input data/hiv.csv -c src/configs/config_regression_hiv.py --output tmp/ --feature maccs
python src/scripts/regression.py --input data/hiv.csv -c src/configs/config_regression_hiv.py --output tmp/ --feature morgan --nBits 512
python src/scripts/regression.py --input data/hiv.csv -c src/configs/config_regression_hiv.py --output tmp/ --feature morgan --nBits 1024
python src/scripts/regression.py --input data/hiv.csv -c src/configs/config_regression_hiv.py --output tmp/ --feature morgan --nBits 2048

python src/scripts/residual.py --input data/hiv.csv -c src/configs/config_residual_hiv.py --output tmp/ --feature maccs
python src/scripts/residual.py --input data/hiv.csv -c src/configs/config_residual_hiv.py --output tmp/ --feature morgan --nBits 512
python src/scripts/residual.py --input data/hiv.csv -c src/configs/config_residual_hiv.py --output tmp/ --feature morgan --nBits 1024
python src/scripts/residual.py --input data/hiv.csv -c src/configs/config_residual_hiv.py --output tmp/ --feature morgan --nBits 2048
