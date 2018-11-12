import sys
import numpy as np
import pandas as pd
# from sklearn.preprocessing import Imputer
from moloi.create_model import create_model, create_gridsearch_model
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from moloi.config_processing import read_model_config
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
# from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA, TruncatedSVD
from moloi.splits.cv import create_cv


def create_pipeline(logger, options, exp_settings, scoring, data, gridsearch=False):
    # minmax_y for regression problem
    # minmax for svm
    steps = options.select_model
    # pipe = [('imputation', Imputer(missing_values='NaN', strategy = 'most_frequent', axis=0))]
    pipe = []
    classifiers= []
    # labels processing
    pipe_y = []
    if 'minmax_y' in steps:
        steps.remove('minmax_y')
        pipe_y.append(('minmax_y', MinMaxScaler()))

    if len(pipe_y) > 1:
        pipeline_y = Pipeline([pipe_y])
        pipeline_y.fit(data['y_train'])
        data['y_train'] = pipeline_y.transform(data['y_train'])
        data['y_test'] = pipeline_y.transform(data['y_test'])
        data['y_val'] = pipeline_y.transform(data['y_val'])
    if len(pipe_y) == 1:
        pipe_y[0][1].fit(data['y_train'])
        data['y_train'] = pipe_y[0][1].transform(data['y_train'])
        data['y_test'] = pipe_y[0][1].transform(data['y_test'])
        data['y_val'] = pipe_y[0][1].transform(data['y_val'])

    # preprocessing
    if 'dummies' in steps:
        data['x_train'] = np.array(pd.get_dummies(pd.DataFrame(data['x_train'])))
        data['x_test'] = np.array(pd.get_dummies(pd.DataFrame(data['x_test'])))
        data['x_val'] = np.array(pd.get_dummies(pd.DataFrame(data['x_val'])))
    
    if 'minmax' in steps:
        _, rparams, _ = read_model_config(options.model_config, 'minmax')
        pipeline_x = MinMaxScaler(**rparams)
        
    if 'norm' in steps:
        _, rparams, _ = read_model_config(options.model_config, 'norm')
        pipeline_x = Normalizer(**rparams)
        
    if 'ss' in steps:
        _, rparams, _ = read_model_config(options.model_config, 'ss')
        pipeline_x = StandardScaler(**rparams)
        
    if 'minmax' in steps or 'norm' in steps or 'ss' in steps:
        if 'minmax' in steps:
            steps.remove('minmax')
        if 'norm' in steps:
            steps.remove('norm')
        if 'ss' in steps:
            steps.remove('ss')
        pipeline_x.fit(data['x_train'])
        data['x_train'] = pipeline_x.transform(data['x_train'])
        data['x_test'] = pipeline_x.transform(data['x_test'])
        data['x_val'] = pipeline_x.transform(data['x_val'])

    # imbalanced
    if 'ros' in steps:
        _, rparams, _ = read_model_config(options.model_config, 'ros')
        imbalanced = RandomOverSampler(random_state=exp_settings["random_state"], **rparams)

    if 'ada' in steps:
        _, rparams, _ = read_model_config(options.model_config, 'ada')
        imbalanced = ADASYN(random_state=exp_settings["random_state"], **rparams)

    if 'smote' in steps:
        _, rparams, _ = read_model_config(options.model_config, 'smote')
        imbalanced = SMOTE(random_state=exp_settings["random_state"], **rparams)

    if 'smoteenn' in steps:
        _, rparams, _ = read_model_config(options.model_config, 'smoteenn')
        imbalanced = SMOTEENN(random_state=exp_settings["random_state"], **rparams)

    if 'smotetomek' in steps:
        _, rparams, _ = read_model_config(options.model_config, 'smotetomek')
        imbalanced = SMOTETomek(random_state=exp_settings["random_state"], **rparams)
    
    if 'ros' in steps or 'ada' in steps or 'smote' in steps or 'smoteenn' in steps or 'smotetomek' in steps:
        if 'ros' in steps:
            steps.remove('ros')
        if 'ada' in steps:
            steps.remove('ada')
        if 'smote' in steps:
            steps.remove('smote')
        if 'smoteenn' in steps:
            steps.remove('smoteenn')
        if 'smotetomek' in steps:
            steps.remove('smotetomek')
        data['x_train'], data['y_train'] = imbalanced.fit_resample(data['x_train'], data['y_train'])
        data['x_test'], data['y_test'] = imbalanced.fit_resample(data['x_test'], data['y_test'])
        data['x_val'], data['y_val'] = imbalanced.fit_resample(data['x_val'], data['y_val'])
        data['y_train'] = np.array(data['y_train']).reshape(-1,1)
        data['y_test'] = np.array(data['y_test']).reshape(-1,1)
        data['y_val'] = np.array(data['y_val']).reshape(-1,1)
    
    # feature_selection/decomposition
    preprocessors = []
    if 'pca' in steps:
        steps.remove('pca')
        _, rparams, _ = read_model_config(options.model_config, 'pca')
        pca = PCA(**rparams)
        preprocessors.append(('pca', pca))
        
    if 'selectkbest' in steps:
        steps.remove('selectkbest')
        _, rparams, _ = read_model_config(options.model_config, 'selectkbest')
        selection = SelectKBest(**rparams)
        preprocessors.append(('selectkbest', selection))
        
    if 'tsvd' in steps:
        steps.remove('tsvd')
        _, rparams, _ = read_model_config(options.model_config, 'tsvd')
        tsvd = TruncatedSVD(**rparams)
        preprocessors.append(('tsvd', tsvd))

    if len(preprocessors) > 1:
        pipe.append(('preprocessors', FeatureUnion(preprocessors)))
    elif len(preprocessors) == 1:
        pipe.append(preprocessors[0])

    models_gparams = False
    for s in steps:
        epochs, rparams, gparams = read_model_config(options.model_config, s)
        if gridsearch:# and len(steps) >= 1:
            t_gparams = dict(gparams)
            
            if len(steps) < 1:
                print("There are no estimators")
                sys.exit(0)
            if len(pipe) == 0 and len(steps) == 1:
                pass # clean model
            if len(pipe) >= 1 and len(steps) == 1:
                for g in t_gparams:
                    gparams[s+'__'+g] = gparams.pop(g) # pipe with vote # pipe without vote
            if len(pipe) >= 0 and len(steps) > 1:
                for g in t_gparams:
                    gparams['voting__'+s+'__'+g] = gparams.pop(g) # pipe with vote
           
            """
            if len(pipe) >= 0:
                for g in t_gparams:
                    if len(steps) > 1:
                        gparams['voting__'+s+'__'+g] = gparams.pop(g)
                    #else:
                    #    gparams[s+'__'+g] = gparams.pop(g)
            else:
                for g in t_gparams:
                    if len(steps) > 1:
                        gparams[s+'__'+g] = gparams.pop(g)
            """
            if not models_gparams:
                models_gparams = gparams
            else:
                models_gparams.update(gparams)

        keys = list(gparams.keys())
        n_iter = 1
        for k in keys:
            n_iter *= len(gparams[k])
        if options.n_iter > n_iter:
            options.n_iter = n_iter

        if '_keras' in s and gridsearch:
            step_model = create_gridsearch_model(logger, rparams, gparams, options, exp_settings, scoring, s, data["x_train"].shape[1], data["y_train"].shape[1])
        else:
            step_model = create_model(logger, rparams, options, s, data["x_train"].shape[1], data["y_train"].shape[1])
            
        classifiers.append((s, step_model))

    if models_gparams:
        gparams = dict(models_gparams)

    Vote = False
    if len(steps) < 2:
        pipe.append(classifiers[0])
    else:
        pipe.append(("voting", VotingClassifier(classifiers, voting='soft')))
        Vote = True

    if len(pipe) > 1 or Vote:
        print("Create pipe")
        model = Pipeline(pipe)
    elif len(pipe) == 1:
        print("Create model")
        model = pipe[0][1]
    else:
        print("No model loaded")
        sys.exit(0)

    if not '_keras' in s and gridsearch:
        sklearn_params = {
            'param_distributions': gparams,
            'n_iter': options.n_iter,
            'n_jobs': options.n_jobs,
            'verbose': exp_settings["verbose"],
            'scoring': scoring,
            'cv': options.n_cv,
            'return_train_score': True,
            'refit': True,
            'random_state': exp_settings["random_state"]}
        
        model = RandomizedSearchCV(estimator=model, **sklearn_params)
    
    print(model)
    import pprint as pp
    pp.pprint(sorted(model.get_params().keys()))
    print(gparams)
    #sys.exit(0)

    return model, epochs, data, rparams
