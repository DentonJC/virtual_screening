# Virtual screening

Run scripts as:

```
 python src/scripts/regression.py --input data/preprocessed/HIV_maccs.csv -c src/configs/config_regression_hiv.py --output tmp/ --feature maccs -g
```
### Usage
Usage : script.py -i <input_file> -c <config_file> -f <featurizer> -o <output_file> -g (grid_search) -d (dummy_data)

or

script.py --input <input_file> --config <config_file> --feature <featurizer> --output <output_file> -g (grid_search) -d (dummy_data)

Warning: With custom accuracy, model checkpoints do not work.

## Citation
O. Tange (2011): GNU Parallel - The Command-Line Power Tool,  ;login: The USENIX Magazine, February 2011:42-47.
