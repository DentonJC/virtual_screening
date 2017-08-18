# Virtual screening

Run scripts as:

```
 python src/scripts/regression_hiv.py --input data/preprocessed/HIV_maccs.csv --output tmp --feature maccs --dummy True
```
### Usage
script.py -i <input_file> -f <featurizer> -o <output_file> -g (grid_search) -d (dummy_data)

or

script.py --input <input_file> --feature <featurizer> --output <output_file> --grid (grid_search) --dummy (dummy_data)

## Citation
O. Tange (2011): GNU Parallel - The Command-Line Power Tool,  ;login: The USENIX Magazine, February 2011:42-47.
