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
        'accuracy_test': False,
        'accuracy_train': False,
        'accuracy_val': False,
        'rec_train': False,
        'rec_test': False,
        'rec_val': False,
        'auc_test': False,
        'auc_train': False,
        'auc_val': False,
        'f1_train': False,
        'f1_test': False,
        'f1_val': False,
        'rmse_test': False,
        'rmse_train': False,
        'rmse_val': False,
        'mae_test': False,
        'mae_train': False,
        'mae_val': False,
        'r2_test': False,
        'r2_train': False,
        'r2_val': False,
        'rparams': False
        }
    return results
