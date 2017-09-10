n_folds = 5 #grid_search
assert (n_folds>=3) and (n_folds<=10), "Looks like the number of folds is not in between 3 and 10"

epochs = 100000

#Choose target
targets = [3]
#output_shape = output_shape
#Chose features
features = False
#input_shape = input_shape

rparams = {
    "batch_size": 1000, 
    "activation": 'sigmoid', 
    "optimizer": 'Adam', 
    "loss": 'binary_crossentropy',    
    #"neurons": 32, 
    "learning_rate": .000001,     
    "momentum": .9,
    "init_mode": 'uniform',
    "metrics": ['accuracy']
    }

# For greed search
gparams = {
    "epochs" : [5],
    "batch_size": [8, 32, 128, 512, 2048],
    "activation": ['softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
    "optimizer": ['Adam', 'RMSprop', 'SGD', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'],
    "loss": ['categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 
            'mean_squared_logarithmic_error', 'squared_hinge', 'hinge','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity'],
    # "neurons": [8], #[8, 16, 32, 64],
    "learning_rate": [0.1, .001, .0001, .00001],
    "momentum": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
    "init_mode": ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    }

"""
gparams = {
    "epochs" : [2],
    "batch_size": [128],
    "activation": ['relu'],
    "optimizer": ['SGD'],
    "loss": ['categorical_crossentropy'],
    # "neurons": [8], #[8, 16, 32, 64],
    "learning_rate": [0.1, .001, .0001, .00001],
    "momentum": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
    "init_mode": ['uniform']
    }
"""

