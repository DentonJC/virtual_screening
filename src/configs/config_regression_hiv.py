epochs = 1000

#Choose target
targets = False
#output_shape = output_shape
#Chose features
features = False
#input_shape = input_shape

rparams = {
    "batch_size": 128, 
    "activation": 'sigmoid', 
    "optimizer": 'Adam', 
    "loss": 'binary_crossentropy',    
    "learning_rate": .1,    
    "momentum": 0,
    "init_mode": 'he_uniform',
    "metrics": ['accuracy']
    }

# For greed search
"""
gparams = {
    "epochs" : [1],
    "batch_size": [8, 32, 128, 512, 2048],
    "activation": ['softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
    "optimizer": ['Adam', 'RMSprop', 'SGD', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'],
    "loss": ['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 
            'squared_hinge', 'hinge','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity'],
    ## "neurons": [8, 16, 32, 64],
    "learning_rate": [0.1, .001, .0001, .00001],
    "momentum": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
    "init_mode": ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    }
"""
"""
"loss": ['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 
            'squared_hinge', 'hinge','binary_crossentropy','kullback_leibler_divergence','poisson'],
"""
gparams = {
    "epochs" : [1],
    "batch_size": [8, 16, 32, 64, 128],
    "activation": ['softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
    "optimizer": ['Adam'],
    "loss": ['binary_crossentropy'],
    ## "neurons": [8, 16, 32, 64],
    "learning_rate": [0.1],
    "momentum": [0.0],
    "init_mode": ['he_uniform']
    }
