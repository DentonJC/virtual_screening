def addresses_dict():
    addresses = {
        'dataset_train': False,
        'dataset_test': False,
        'dataset_val': False,
        'labels_train': False,
        'labels_test': False,
        'labels_val': False,
        'maccs_train': False,
        'maccs_test': False,
        'maccs_val': False,
        'morgan_train': False,
        'morgan_test': False,
        'morgan_val': False,
        'spectrophore_train': False,
        'spectrophore_test': False,
        'spectrophore_val': False,
        'mordred_train': False,
        'mordred_test': False,
        'mordred_val': False,
        'rdkit_train': False,
        'rdkit_test': False,
        'rdkit_val': False,
        'external_train': False,
        'external_test': False,
        'external_val': False
        }
    return addresses


def data_dict():
    data = {
        'x_train': False,
        'x_test': False,
        'x_val': False,
        'y_train': False,
        'y_test': False,
        'y_val': False,

        'smiles_train': False,
        'labels_train': False,
        'full_smiles_train': False,

        'smiles_test': False,
        'labels_test': False,
        'full_smiles_test': False,

        'smiles_val': False,
        'labels_val': False,
        'full_smiles_val': False
        }
    return data


def results_dict():
    results = {
        'accuracy_test': '-',
        'accuracy_train': '-',
        'rec': '-',
        'auc': '-',
        'auc_val': '-',
        'f1': '-',
        'rparams': '-',
        'model_address': '-',
        'balanced_accuracy': '-'
        }
    return results
