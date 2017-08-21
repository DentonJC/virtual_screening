patience = 20
epochs = 1000

rparams = {
    "batch_size": 32, 
    "activation": 'softmax', 
    "optimizer": 'Adam', 
    "loss": 'categorical_crossentropy',    
    "neurons": 32, 
    "learning_rate": .001,     
    "momentum": .1,
    "init_mode": 'uniform',
    "metrics": ['accuracy']
    }

# For greed search
gparams = {
    "epochs" : [3],
    "batch_size": [8, 16, 32, 64, 128],
    "activation": ['softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
    "optimizer": ['Adam', 'RMSprop', 'SGD', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'],
    "loss": ['categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 
            'mean_squared_logarithmic_error', 'squared_hinge', 'hinge','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity'],
    # "neurons": [8], #[8, 16, 32, 64],
    "learning_rate": [0.1, .001, .0001, .00001],
    "momentum": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
    "init_mode": ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    }
