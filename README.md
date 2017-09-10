# Virtual screening

Run scripts as:

```
 python src/scripts/regression.py --input data/muv.csv -c src/configs/config_regression_muv.py --output tmp/ --feature morgan --nBits 2048 -g
```
### Usage

Usage : script.py -i <input_file> -c <config_file> -f <featurizer> -n <number_of_bits> -o <output_file> -g (grid_search) -d (dummy_data)

or

script.py --input <input_file> --config <config_file> --feature <featurizer> --nBits <number_of_bits> --output <output_file> -g (grid_search) -d (dummy_data))

Warning: With custom accuracy, model checkpoints do not work.

## Citation
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
- Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array: A Structure for Efficient Numerical Computation, Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37 (publisher link)
- Travis E. Oliphant. Python for Scientific Computing, Computing in Science & Engineering, 9, 10-20 (2007), DOI:10.1109/MCSE.2007.58 (publisher link)
- K. Jarrod Millman and Michael Aivazis. Python for Scientists and Engineers, Computing in Science & Engineering, 13, 9-12 (2011), DOI:10.1109/MCSE.2011.36 (publisher link)
- Fernando Pérez and Brian E. Granger. IPython: A System for Interactive Scientific Computing, Computing in Science & Engineering, 9, 21-29 (2007), DOI:10.1109/MCSE.2007.53 (publisher link)
- John D. Hunter. Matplotlib: A 2D Graphics Environment, Computing in Science & Engineering, 9, 90-95 (2007), DOI:10.1109/MCSE.2007.55 (publisher link)
- Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010) (publisher link)
- O. Tange (2011): GNU Parallel - The Command-Line Power Tool,  ;login: The USENIX Magazine, February 2011:42-47.
- https://github.com/gmum/ananas/blob/master/fingerprints/_desc_rdkit.py
- RDKit: Open-source cheminformatics; http://www.rdkit.org
- Keras (2015), Chollet et al., https://github.com/fchollet/keras
