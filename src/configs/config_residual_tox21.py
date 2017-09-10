n_folds = 3 #grid_search
assert (n_folds>=3) and (n_folds<=10), "Looks like the number of folds is not in between 3 and 10"

epochs = 1000

#Choose target
targets = False
#output_shape = output_shape
#Chose features
features = False
#input_shape = input_shape

rparams = {
    "batch_size": 64, 
    "activation_0": 'relu', 
    "activation_1": 'relu', 
    "activation_2": 'relu',
    "optimizer": 'Adam', 
    "loss": 'categorical_crossentropy',    
    # "neurons": 32, 
    "learning_rate": .001,     
    "momentum": .1,
    "init_mode": 'uniform',
    "metrics": ['accuracy'],
    "dropout": 0.1,
    "layers": 3
    }

# For greed search
gparams = {
    "epochs" : [5],
    "batch_size": [8, 32, 128, 512, 2048],
    "activation_0": ['softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
    "activation_1": ['softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
    "activation_2": ['softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
    "optimizer": ['Adam', 'RMSprop', 'SGD', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'],
    "loss": ['categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 
            'mean_squared_logarithmic_error', 'squared_hinge', 'hinge','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity'],
    # "neurons": [8], #[8, 16, 32, 64],
    "learning_rate": [.01, .001, .0001, .00001],
    "momentum": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
    "init_mode": ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
    "dropout": [0,0.2,0.5,0.7],
    "layers": [1,2,3]
    }
