import sys
from sklearn.svm import SVC, SVR
# from sklearn.utils import class_weight as cw
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # , PredefinedSplit
from keras.wrappers.scikit_learn import KerasClassifier
from moloi.models.keras_models import FCNN, LSTM, MLP, Logreg
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from keras.wrappers.scikit_learn import KerasClassifier
import xgboost as xgb


def create_gridsearch_model(logger, rparams, gparams, options, exp_settings, scoring, select_model, input_shape, output_shape):
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
        'n_iter': options.n_iter,
        'n_jobs': options.n_jobs,
        'cv': options.n_cv,
        'verbose': exp_settings["verbose"],
        'scoring': scoring,
        'return_train_score': True,
        'refit': True,
        'pre_dispatch': options.n_cv,
        'random_state': exp_settings["random_state"]}

    grid_sklearn_params = {
        'param_grid': gparams,
        'n_jobs': options.n_jobs,
        'cv': options.n_cv,
        'verbose': exp_settings["verbose"],
        'scoring': scoring,
        'return_train_score': True,
        'refit': True}

    # sklearn models
    if select_model == "lr":
        model = RandomizedSearchCV(LogisticRegression(**rparams), **sklearn_params)
    elif select_model == "knn":
        model = RandomizedSearchCV(KNeighborsClassifier(**rparams), **sklearn_params)
    elif select_model == "xgb":
        model = RandomizedSearchCV(xgb.XGBClassifier(**rparams), **sklearn_params)
    elif select_model == "svc":
        if type(gparams) == list:
            model = GridSearchCV(SVC(**rparams, probability=True), **grid_sklearn_params)
        else:
            model = RandomizedSearchCV(SVC(**rparams, probability=True), **sklearn_params)
    elif select_model == "rf":
        model = RandomizedSearchCV(RandomForestClassifier(**rparams), **sklearn_params)
    elif select_model == "et":
        model = RandomizedSearchCV(ExtraTreesClassifier(**rparams), **sklearn_params)
    elif select_model == "if":
        model = RandomizedSearchCV(IsolationForest(**rparams), **sklearn_params)
    elif select_model == "rbm":
        model = RandomizedSearchCV(BernoulliRBM(**rparams), **sklearn_params)
    elif select_model == "mlp_sklearn":
        model = RandomizedSearchCV(MLPClassifier(**rparams), **sklearn_params)
    # keras models
    elif select_model == "regression":
        search_model = KerasClassifier(build_fn=Logreg, input_shape=input_shape, output_shape=output_shape)
        model = RandomizedSearchCV(estimator=search_model, **keras_params)
    elif select_model == "fcnn":
        search_model = KerasClassifier(build_fn=FCNN, input_shape=input_shape, output_shape=output_shape)
        model = RandomizedSearchCV(estimator=search_model, **keras_params)
    elif select_model == "lstm":
        search_model = KerasClassifier(build_fn=LSTM, input_shape=input_shape,
                                       output_shape=output_shape, input_length=input_shape)
        model = RandomizedSearchCV(estimator=search_model, **keras_params)
    elif select_model == "mlp_keras":
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
    elif select_model == "lr_reg":
        pol_features = PolynomialFeatures()
        lr = LinearRegression()
        pipeline = Pipeline([('pol_features', pol_features), ("regression", lr)])
        model = RandomizedSearchCV(pipeline, **sklearn_params)
    elif select_model == "knn_reg":
        model = RandomizedSearchCV(KNeighborsRegressor(**rparams), **sklearn_params)
    elif select_model == "rf_reg":
        model = RandomizedSearchCV(RandomForestRegressor(**rparams), **sklearn_params)
    elif select_model == "ir":
        model = RandomizedSearchCV(IsotonicRegression(**rparams), **sklearn_params)
    elif select_model == "xgb_reg":
        model = RandomizedSearchCV(xgb.XGBRegressor(**rparams), **sklearn_params)
    elif select_model == "lgbmr":
        model = RandomizedSearchCV(LGBMRegressor(**rparams), **sklearn_params)
    elif select_model == "lgbmc":
        model = RandomizedSearchCV(LGBMClassifier(**rparams), **sklearn_params)
        
    elif select_model == "abc":
        model = RandomizedSearchCV(AdaBoostClassifier(**rparams), **sklearn_params)
    elif select_model == "abr":
        model = RandomizedSearchCV(AdaBoostRegressor(**rparams), **sklearn_params)
    elif select_model == "bagc":
        model = RandomizedSearchCV(BaggingClassifier(**rparams), **sklearn_params)
    elif select_model == "bagr":
        model = RandomizedSearchCV(BaggingRegressor(**rparams), **sklearn_params)
    elif select_model == "gbc":
        model = RandomizedSearchCV(GradientBoostingClassifier(**rparams), **sklearn_params)
    elif select_model == "gbr":
        model = RandomizedSearchCV(GradientBoostingRegressor(**rparams), **sklearn_params)
    elif select_model == "dtc":
        model = RandomizedSearchCV(DecisionTreeClassifier(**rparams), **sklearn_params)
    elif select_model == "dtr":
        model = RandomizedSearchCV(DecisionTreeRegressor(**rparams), **sklearn_params)
    elif select_model == "etc":
        model = RandomizedSearchCV(ExtraTreeClassifier(**rparams), **sklearn_params)
    elif select_model == "etr":
        model = RandomizedSearchCV(ExtraTreeRegressor(**rparams), **sklearn_params)
    elif select_model == "etsc":
        model = RandomizedSearchCV(ExtraTreesClassifier(**rparams), **sklearn_params)
    elif select_model == "etsr":
        model = RandomizedSearchCV(ExtraTreesRegressor(**rparams), **sklearn_params)

    elif select_model == "svr":
        if type(gparams) == list:
            model = GridSearchCV(SVR(**rparams), **grid_sklearn_params)
        else:
            model = RandomizedSearchCV(SVR(**rparams), **sklearn_params)
    elif select_model == "mlp_sklearn_reg":
        model = RandomizedSearchCV(MLPRegressor(**rparams), **sklearn_params)
    else:
        logger.info("Model name is not found.")
        sys.exit(0)

    return model

def create_model(logger, rparams, options, select_model, input_shape, output_shape):
    keras_rparams = dict(rparams)
    if 'epochs' in keras_rparams:
        del keras_rparams['epochs']
    if 'class_weight' in keras_rparams:
        del keras_rparams['class_weight']
    if 'batch_size' in keras_rparams:
        del keras_rparams['batch_size']
        
    if 'voting__epochs' in keras_rparams:
        del keras_rparams['voting__epochs']
    if 'voting__class_weight' in keras_rparams:
        del keras_rparams['voting__class_weight']
    if 'voting__batch_size' in keras_rparams:
        del keras_rparams['voting__batch_size']

    if select_model == "lr":
        model = LogisticRegression(**rparams)
    elif select_model == "knn":
        model = KNeighborsClassifier(**rparams)
    elif select_model == "xgb":
        model = xgb.XGBClassifier(**rparams)
    elif select_model == "svc":
        model = SVC(**rparams, probability=True)
    elif select_model == "rf":
        model = RandomForestClassifier(**rparams)
    elif select_model == "et":
        model = RandomForestClassifier(**rparams)
    elif select_model == "if":
        model = IsolationForest(**rparams)
    elif select_model == "rbm":
        model = BernoulliRBM(**rparams)
    elif select_model == "mlp_sklearn":
        model = MLPClassifier(**rparams)

    elif select_model == "fcnn":
        model = FCNN(input_shape, output_shape, **keras_rparams)
    elif select_model == "regression":
        model = Logreg(input_shape, output_shape, **keras_rparams)
    elif select_model == "mlp_keras":
        # model = MLP(input_shape, output_shape, **keras_rparams)
        model = KerasClassifier(build_fn=MLP, input_shape=input_shape, output_shape=output_shape)
    # elif options.select_model == "rnn":
    #     model = RNN(input_shape, output_shape, **rparams)
    # elif options.select_model == "gru":
    #     model = GRU(input_shape, output_shape, **rparams)
    # elif options.select_model == "lstm":
    #     model = LSTM(input_shape, output_shape, input_length=x_train.shape[1], **rparams)
    elif select_model == "lr_reg":
        pol_features = PolynomialFeatures()
        lr = LinearRegression()
        pipeline = Pipeline([('pol_features', pol_features), ("regression", lr)])
        pipeline.set_params(**rparams)
        model = pipeline
    elif select_model == "knn_reg":
        model = KNeighborsRegressor(**rparams)
    elif select_model == "ir":
        model = IsotonicRegression(**rparams)
    elif select_model == "rf_reg":
        model = RandomForestRegressor(**rparams)
    elif select_model == "xgb_reg":
        model = xgb.XGBRegressor(**rparams)
    elif select_model == "svr":
        model = SVR(**rparams)
    elif select_model == "mlp_sklearn_reg":
        model = MLPRegressor(**rparams)
    elif select_model == "lgbmr":
        model = LGBMRegressor(**rparams)
    elif select_model == "lgbmc":
        model = LGBMClassifier(**rparams)
    
    elif select_model == "abc":
        model = AdaBoostClassifier(**rparams)
    elif select_model == "abr":
        model = AdaBoostRegressor(**rparams)
    elif select_model == "bagc":
        model = BaggingClassifier(**rparams)
    elif select_model == "bagr":
        model = BaggingRegressor(**rparams)
    elif select_model == "gbc":
        model = GradientBoostingClassifier(**rparams)
    elif select_model == "gbr":
        model = GradientBoostingRegressor(**rparams)
    elif select_model == "dtc":
        model = DecisionTreeClassifier(**rparams)
    elif select_model == "dtr":
        model = DecisionTreeRegressor(**rparams)
    elif select_model == "etc":
        model = ExtraTreeClassifier(**rparams)
    elif select_model == "etr":
        model = ExtraTreeRegressor(**rparams)
    elif select_model == "etsc":
        model = ExtraTreesClassifier(**rparams)
    elif select_model == "etsr":
        model = ExtraTreesRegressor(**rparams)
    else:
        logger.info("Model name is not found or xgboost import error.")
        sys.exit(0)

    return model
