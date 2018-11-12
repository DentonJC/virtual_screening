#!/usr/bin/env python

import os
# import pickle
# import logging
from sklearn.externals import joblib
from keras.models import model_from_json
import zipfile


def load_model(load_model, logger):
    model_loaded, rparams, model = False, False, False
    try:
        if type(eval(load_model)) is int or type(eval(load_model)) is float:
            fp = open(os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp/addresses")
            for i, line in enumerate(fp):
                if i == eval(load_model) - 1:
                    load_model = eval(line)
                    break
            fp.close()
        else:
            load_model = eval(load_model)
            
        zip_ref = zipfile.ZipFile(os.path.dirname(load_model[0]) + '.zip', 'r')
        zip_ref.extractall(os.path.dirname(load_model[0]))
        zip_ref.close()

        if load_model[1] is False:
            # model = pickle.load(open(options.load_model, 'rb'))
            model = joblib.load(load_model[0])
        else:
            # load json and create model
            json_file = open(load_model[0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(load_model[1])
            compile_params = [line.rstrip('\n') for line in open(load_model[2])]
            model.compile(loss=compile_params[0], metrics=eval(compile_params[1]), optimizer=compile_params[2])
            rparams = eval(compile_params[3])
        model_loaded = True
        logger.info("Model loaded")
    except:
        model_loaded = False
        logger.info("Model not loaded")
    return model, rparams, model_loaded


def save_model(model, path, logger, rparams):
    model_address = False
    try:
        # pickle.dump(model, open(options.output + "model.sav", 'wb'))
        joblib.dump(model, path + "model.sav")
        model_address = [path + "model.sav", False, False, str(rparams)]
        f = open(path + 'addresses', 'w')
        f.write(str(model_address))
        f.close()

        f = open(os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp/addresses", 'a')
        f.write(str(model_address) + '\n')
        f.close()
        model_address = sum(1 for line in open(os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") +
                                               "/tmp/addresses"))

    except:
        try:
            # serialize model to JSON
            model_json = model.to_json()
            with open(path + "model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(path + "model.h5")

            loss = rparams.get("loss", 'binary_crossentropy')
            metrics = rparams.get("metrics", ['accuracy'])
            optimizer = rparams.get("optimizer", 'Adam')
            """
            if "loss" in rparams:
                del rparams["loss"]
            if "metrics" in rparams:
                del rparams["metrics"]
            if "optimizer" in rparams:
                del rparams["optimizer"]
            """
            f = open(path + 'compile', 'w')
            f.write(loss + '\n' + str(metrics) + '\n' + optimizer + '\n' + str(rparams))
            f.close()
            model_address = [path + "model.json", path + "model.h5", path + 'compile']
            f = open(path + 'addresses', 'w')
            f.write(str(model_address))
            f.close()

            f = open(os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") + "/tmp/addresses", 'a')
            f.write(str(model_address) + '\n')
            f.close()
            model_address = sum(1 for line in open(os.path.dirname(os.path.realpath(__file__)).replace("/moloi", "") +
                                                   "/tmp/addresses"))
        except:
            logger.info("Can not save this model")
    return model_address
