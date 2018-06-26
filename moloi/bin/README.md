# User scripts

To create user script using moloi as a library, you can import the following modules:
 - <b>from moloi.config_processing import read_model_config</b> - to load the hyperparameters search grid
 - <b>from moloi.evaluation import evaluate, make_scoring</b> - to create plots, report and get the result of the model evaluation
 - <b>from moloi.splits.cv import create_cv</b> - to create a non-random split
 - <b>from moloi.data_processing import get_data, clean_data</b> - to extract features from SMILES using prepared descriptors
 - <b>from moloi.models.keras_models import Logreg, create_callbacks</b> - to use one of the ready-made Keras models
