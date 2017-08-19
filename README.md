# Virtual screening

Run scripts as:

```
 python src/scripts/regression_hiv.py --input data/HIV.csv --output tmp/ --feature maccs -g
```
### Usage
script.py -i <input_file> -f <featurizer> -o <output_file> -g (grid_search) -d (dummy_data)

or

script.py --input <input_file> --feature <featurizer> --output <output_file> -g (grid_search) -d (dummy_data)

Warning: With custom accuracy, model checkpoints do not work.

## Citation
O. Tange (2011): GNU Parallel - The Command-Line Power Tool,  ;login: The USENIX Magazine, February 2011:42-47.
