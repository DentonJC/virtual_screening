import sys
from sklearn.svm import SVC, SVR
# from sklearn.utils import class_weight as cw
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # , PredefinedSplit
from keras.wrappers.scikit_learn import KerasClassifier
from moloi.models.keras_models import FCNN, LSTM, MLP, Logreg
import xgboost as xgb


def create_gridsearch_model(logger, rparams, gparams, options, exp_settings, scoring, n_cv, input_shape, output_shape):
    sklearn_params = {
        'param_distributions': gparams,
        'n_iter': options.n_iter,
        'n_jobs': options.n_jobs,
        'cv': options.n_cv,
        'verbose': exp_settings["verbose"],
        'scoring': scoring,
        'return_train_score': True,
        'refit': True,
        'random_state': exp_settings["random_state"]}

    keras_params = {
        'param_distributions': gparams,
        'n_jobs': options.n_jobs,
        'cv': options.n_cv,
        'n_iter': options.n_iter,
        'verbose': exp_settings["verbose"],
        'scoring': scoring,
        'random_state': exp_settings["random_state"],
        'return_train_score': True,
        'refit': True,
        'pre_dispatch': n_cv}

    grid_sklearn_params = {
        'param_grid': gparams,
        'n_jobs': options.n_jobs,
        'cv': options.n_cv,
        'verbose': exp_settings["verbose"],
        'scoring': scoring,
        'return_train_score': True,
        'refit': True}

    # sklearn models
    if options.select_model == "lr":
        model = RandomizedSearchCV(LogisticRegression(**rparams), **sklearn_params)
    elif options.select_model == "knn":
        model = RandomizedSearchCV(KNeighborsClassifier(**rparams), **sklearn_params)
    elif options.select_model == "xgb":
        model = RandomizedSearchCV(xgb.XGBClassifier(**rparams), **sklearn_params)
    elif options.select_model == "svc":
        if type(gparams) == list:
            model = GridSearchCV(SVC(**rparams, probability=True), **grid_sklearn_params)
        else:
            model = RandomizedSearchCV(SVC(**rparams, probability=True), **sklearn_params)
    elif options.select_model == "rf":
        model = RandomizedSearchCV(RandomForestClassifier(**rparams), **sklearn_params)
    elif options.select_model == "if":
        model = RandomizedSearchCV(IsolationForest(**rparams), **sklearn_params)
    elif options.select_model == "rbm":
        model = RandomizedSearchCV(BernoulliRBM(**rparams), **sklearn_params)
    elif options.select_model == "mlp_sklearn":
        model = RandomizedSearchCV(MLPClassifier(**rparams), **sklearn_params)
    # keras models
    elif options.select_model == "regression":
        search_model = KerasClassifier(build_fn=Logreg, input_shape=input_shape, output_shape=output_shape)
        model = RandomizedSearchCV(estimator=search_model, **keras_params)
    elif options.select_model == "fcnn":
        search_model = KerasClassifier(build_fn=FCNN, input_shape=input_shape, output_shape=output_shape)
        model = RandomizedSearchCV(estimator=search_model, **keras_params)
    elif options.select_model == "lstm":
        search_model = KerasClassifier(build_fn=LSTM, input_shape=input_shape,
                                       output_shape=output_shape, input_length=input_shape)
        model = RandomizedSearchCV(estimator=search_model, **keras_params)
    elif options.select_model == "mlp":
        search_model = KerasClassifier(build_fn=MLP, input_shape=input_shape, output_shape=output_shape)
        model = RandomizedSearchCV(estimator=search_model, **keras_params)
    # elif options.select_model == "rnn":
    #     search_model = KerasClassifier(build_fn=RNN, input_shape=input_shape,
    #                                    output_shape=output_shape, input_length=x_train.shape[1])
    #     model = RandomizedSearchCV(estimator=search_model, **keras_params)
    # elif options.select_model == "gru":
    #     search_model = KerasClassifier(build_fn=GRU, input_shape=input_shape,
    #                                    output_shape=output_shape, input_length=x_train.shape[1])
    #     model = RandomizedSearchCV(estimator=search_model, **keras_params)
    if options.select_model == "lr_reg":
        model = RandomizedSearchCV(LinearRegression(**rparams), **sklearn_params)
    elif options.select_model == "knn_reg":
        model = RandomizedSearchCV(KNeighborsRegressor(**rparams), **sklearn_params)
    elif options.select_model == "rf_reg":
        model = RandomizedSearchCV(RandomForestRegressor(**rparams), **sklearn_params)
    elif options.select_model == "xgb_reg":
        model = RandomizedSearchCV(xgb.XGBRegressor(**rparams), **sklearn_params)
    elif options.select_model == "svr":
        if type(gparams) == list:
            model = GridSearchCV(SVR(**rparams, probability=True), **grid_sklearn_params)
        else:
            model = RandomizedSearchCV(SVR(**rparams, probability=True), **sklearn_params)
    elif options.select_model == "mlp_sklearn_reg":
        model = RandomizedSearchCV(MLPRegressor(**rparams), **sklearn_params)
    else:
        logger.info("Model name is not found.")
        sys.exit(0)

    return model


def create_model(logger, rparams, options, input_shape, output_shape):
    if options.select_model == "lr":
        model = LogisticRegression(**rparams)
    elif options.select_model == "knn":
        model = KNeighborsClassifier(**rparams)
    elif options.select_model == "xgb":
        model = xgb.XGBClassifier(**rparams)
    elif options.select_model == "svc":
        model = SVC(**rparams, probability=True)
    elif options.select_model == "rf":
        model = RandomForestClassifier(**rparams)
    elif options.select_model == "if":
        model = IsolationForest(**rparams)
    elif options.select_model == "rbm":
        model = BernoulliRBM(**rparams)
    elif options.select_model == "mlp_sklearn":
        model = MLPClassifier(**rparams)

    elif options.select_model == "fcnn":
        model = FCNN(input_shape, output_shape, **rparams)
    elif options.select_model == "regression":
        model = Logreg(input_shape, output_shape, **rparams)
    elif options.select_model == "mlp":
        model = MLP(input_shape, output_shape, **rparams)
    # elif options.select_model == "rnn":
    #     model = RNN(input_shape, output_shape, **rparams)
    # elif options.select_model == "gru":
    #     model = GRU(input_shape, output_shape, **rparams)
    # elif options.select_model == "lstm":
    #     model = LSTM(input_shape, output_shape, input_length=x_train.shape[1], **rparams)
    if options.select_model == "lr_reg":
        model = LinearRegression(**rparams)
    elif options.select_model == "knn_reg":
        model = KNeighborsRegressor(**rparams)
    elif options.select_model == "rf_reg":
        model = RandomForestRegressor(**rparams)
    elif options.select_model == "xgb_reg":
        model = xgb.XGBRegressor(**rparams)
    elif options.select_model == "svr":
        model = SVR(**rparams, probability=True)
    elif options.select_model == "mlp_sklearn_reg":
        model = MLPRegressor(**rparams)
    else:
        logger.info("Model name is not found or xgboost import error.")
        sys.exit(0)

    return model
