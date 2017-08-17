import pickle  
import time  
import json
import sklearn
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def grid_search(param_grid, create_model, x_train, y_train, input_shape, output_shape):
    print("GRID SEARCH")
    search_model = KerasClassifier(build_fn=create_model, input_dim = input_shape, output_dim = output_shape)
    grid = GridSearchCV(estimator=search_model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)
    print(grid_result.best_params_)
    return grid_result.best_params_
