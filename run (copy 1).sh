python src/scripts/regression.py --input data/tox21_morgan_32.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 32 --features 32 --targets 7 -g
python src/scripts/regression.py --input data/tox21_morgan_64.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 64 --features 64 --targets 7 -g
python src/scripts/regression.py --input data/tox21_morgan_128.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 128 --features 128 --targets 7 -g
python src/scripts/regression.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature maccs --n 196 --features 196 --targets 7 -g

python src/scripts/regression.py --input data/muv_morgan_32.csv --config 'src/configs/configs.ini' --section 'REGRESSION_MUV' --output tmp/ --feature morgan --n 32 --features 32 --targets 3 -g
python src/scripts/regression.py --input data/muv_morgan_64.csv --config 'src/configs/configs.ini' --section 'REGRESSION_MUV' --output tmp/ --feature morgan --n 64 --features 64 --targets 3 -g
python src/scripts/regression.py --input data/muv_morgan_128.csv --config 'src/configs/configs.ini' --section 'REGRESSION_MUV' --output tmp/ --feature morgan --n 128 --features 128 --targets 3 -g
python src/scripts/regression.py --input data/preprocessed/muv_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_MUV' --output tmp/ --feature maccs --n 196 --features 196  --targets 3 -g

python src/scripts/regression.py --input data/HIV_morgan_32.csv --config 'src/configs/configs.ini' --section 'REGRESSION_HIV' --output tmp/ --feature morgan --n 32 --features 32 -g
python src/scripts/regression.py --input data/HIV_morgan_64.csv --config 'src/configs/configs.ini' --section 'REGRESSION_HIV' --output tmp/ --feature morgan --n 64 --features 64 -g
python src/scripts/regression.py --input data/HIV_morgan_128.csv --config 'src/configs/configs.ini' --section 'REGRESSION_HIV' --output tmp/ --feature morgan --n 128 --features 128 -g
python src/scripts/regression.py --input data/preprocessed/HIV_maccs.csv --config 'src/configs/configs.ini' --section 'REGRESSION_HIV' --output tmp/ --feature maccs --n 196 --features 196 -g

python src/scripts/regression.py --input data/tox21.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 256 --features 256 --targets 7 -g
python src/scripts/regression.py --input data/tox21.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 512 --features 512 --targets 7 -g
python src/scripts/regression.py --input data/tox21.csv --config 'src/configs/configs.ini' --section 'REGRESSION_TOX21' --output tmp/ --feature morgan --n 1024 --features 1024 --targets 7 -g

python src/scripts/regression.py --input data/muv.csv --config 'src/configs/configs.ini' --section 'REGRESSION_MUV' --output tmp/ --feature morgan --n 256 --features 256 --targets 3 -g
python src/scripts/regression.py --input data/muv.csv --config 'src/configs/configs.ini' --section 'REGRESSION_MUV' --output tmp/ --feature morgan --n 512 --features 512 --targets 3 -g
python src/scripts/regression.py --input data/muv.csv --config 'src/configs/configs.ini' --section 'REGRESSION_MUV' --output tmp/ --feature morgan --n 1024 --features 1024 --targets 3 -g

python src/scripts/regression.py --input data/HIV.csv --config 'src/configs/configs.ini' --section 'REGRESSION_HIV' --output tmp/ --feature morgan --n 256 --features 256 -g
python src/scripts/regression.py --input data/HIV.csv --config 'src/configs/configs.ini' --section 'REGRESSION_HIV' --output tmp/ --feature morgan --n 512 --features 512 -g
python src/scripts/regression.py --input data/HIV.csv --config 'src/configs/configs.ini' --section 'REGRESSION_HIV' --output tmp/ --feature morgan --n 1024 --features 1024 -g






python src/scripts/residual.py --input data/tox21_morgan_32.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature morgan --n 32 --features 32 --targets 7 -g
python src/scripts/residual.py --input data/tox21_morgan_64.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature morgan --n 64 --features 64 --targets 7 -g
python src/scripts/residual.py --input data/tox21_morgan_128.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature morgan --n 128 --features 128 --targets 7 -g
python src/scripts/residual.py --input data/preprocessed/tox21_maccs.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature maccs --n 196 --features 196 --targets 7 -g

python src/scripts/residual.py --input data/muv_morgan_32.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_MUV' --output tmp/ --feature morgan --n 32 --features --targets 3 32 -g
python src/scripts/residual.py --input data/muv_morgan_64.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_MUV' --output tmp/ --feature morgan --n 64 --features --targets 3 64 -g
python src/scripts/residual.py --input data/muv_morgan_128.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_MUV' --output tmp/ --feature morgan --n 128 --features --targets 3 128 -g
python src/scripts/residual.py --input data/preprocessed/muv_maccs.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_MUV' --output tmp/ --feature maccs --n 196 --features 196 --targets 3 -g

python src/scripts/residual.py --input data/HIV_morgan_32.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_HIV' --output tmp/ --feature morgan --n 32 --features 32 -g
python src/scripts/residual.py --input data/HIV_morgan_64.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_HIV' --output tmp/ --feature morgan --n 64 --features 64 -g
python src/scripts/residual.py --input data/HIV_morgan_128.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_HIV' --output tmp/ --feature morgan --n 128 --features 128 -g
python src/scripts/residual.py --input data/preprocessed/HIV_maccs.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_HIV' --output tmp/ --feature maccs --n 196 --features 196 -g



python src/scripts/residual.py --input data/tox21_morgan_256.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature morgan --n 256 --features --targets 7 256 -g
python src/scripts/residual.py --input data/tox21_morgan_512.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature morgan --n 512 --features --targets 7 512 -g
python src/scripts/residual.py --input data/tox21_morgan_1024.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_TOX21' --output tmp/ --feature morgan --n 1024 --features --targets 7 1024 -g

python src/scripts/residual.py --input data/muv_morgan_256.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_MUV' --output tmp/ --feature morgan --n 256 --features 256 --targets 3 -g
python src/scripts/residual.py --input data/muv_morgan_512.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_MUV' --output tmp/ --feature morgan --n 512 --features 512 --targets 3 -g
python src/scripts/residual.py --input data/muv_morgan_1024.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_MUV' --output tmp/ --feature morgan --n 1024 --features 1024 --targets 3 -g

python src/scripts/residual.py --input data/HIV_morgan_256.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_HIV' --output tmp/ --feature morgan --n 256 --features 256 -g
python src/scripts/residual.py --input data/HIV_morgan_512.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_HIV' --output tmp/ --feature morgan --n 512 --features 512 -g
python src/scripts/residual.py --input data/HIV_morgan_1024.csv --config 'src/configs/configs.ini' --section 'RESIDUAL_HIV' --output tmp/ --feature morgan --n 1024 --features 1024 -g