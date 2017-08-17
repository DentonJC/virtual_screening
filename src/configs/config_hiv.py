patience = 1000
epochs = 100

rparams = {
    "batch_size": 64, 
    "activation": 'relu', 
    "optimizer": 'Adam', 
    "loss": 'binary_crossentropy',    
    "neurons": 32, 
    "learning_rate": .001,     
    }

# For greed search
gparams = {
    "epochs" : [3],
    "batch_size": [8, 16, 32, 64, 128],
    "activation": ['softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
    "optimizer": ['Adam', 'RMSprop', 'SGD', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'],
    "loss": ['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity']
    #"neurons": [8], #[8, 16, 32, 64]
    #"learning_rate": [.01] #[.001, .0001, .00001]
    }
