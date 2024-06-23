import numpy as np
import pandas as pd
import gc
import os
from pathlib import Path
from ast import literal_eval
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_score, make_scorer,recall_score,f1_score,roc_auc_score

def hyperparameter_tune(base_model, parameters, kfold, X, Y,groups):
    k = StratifiedKFold(n_splits=kfold, shuffle=False)

    prec_1 = make_scorer(precision_score, pos_label=1)
    rec_1 = make_scorer(recall_score, pos_label=1)
    f1_1 = make_scorer(f1_score, pos_label=1)
    roc = make_scorer(roc_auc_score)

    prec_0 = make_scorer(precision_score, pos_label=0)
    rec_0 = make_scorer(recall_score, pos_label=0)
    f1_0 = make_scorer(f1_score, pos_label=0)

    metrics = {'prec_1': prec_1, 'rec_1': rec_1, 'f1_1': f1_1, 'roc': roc, 'prec_0': prec_0, 'rec_0': rec_0,
               'f1_0': f1_0}

    optimal_model = RandomizedSearchCV(base_model, parameters,scoring=metrics, n_iter=200, cv=k, verbose=3,refit='rec_1', return_train_score=True)
    optimal_model.fit(X, Y,groups)

    return optimal_model.best_params_, optimal_model.best_score_, optimal_model.cv_results_


#data = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_dummies.csv')
#data = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_corine_level2_onehotenc.csv')
data = pd.read_csv('/home/sgirtsou/Documents/datasets/old_random_new_feat_from_months_no_unnamed.csv')
data = data.dropna()

X = data[['max_temp', 'min_temp', 'mean_temp','res_max', 'dom_vel', 'rain_7days', 'dem', 'slope', 'curvature',
       'aspect', 'ndvi_new', 'evi', 'lst_day', 'lst_night', 'max_dew_temp','mean_dew_temp', 'min_dew_temp', 'dir_max_1', 'dir_max_2',
       'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6', 'dir_max_7','dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3', 'dom_dir_4',
       'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8', 'corine_111','corine_112', 'corine_121', 'corine_122', 'corine_123', 'corine_124',
       'corine_131', 'corine_132', 'corine_133', 'corine_141', 'corine_142','corine_211', 'corine_212', 'corine_213', 'corine_221', 'corine_222',
       'corine_223', 'corine_231', 'corine_241', 'corine_242', 'corine_243','corine_244', 'corine_311', 'corine_312', 'corine_313', 'corine_321',
       'corine_322', 'corine_323', 'corine_324', 'corine_331', 'corine_332','corine_333', 'corine_334', 'corine_411', 'corine_412', 'corine_421',
       'corine_422', 'corine_511', 'corine_512', 'corine_521', 'wkd_0','wkd_1', 'wkd_2', 'wkd_3', 'wkd_4', 'wkd_5', 'wkd_6', 'month_5',
       'month_6', 'month_7', 'month_8', 'month_9', 'month_4', 'frequency','f81', 'x', 'y']]
Y = data['fire']

model = ExtraTreesClassifier(n_jobs =8)
groups = data['firedate']
groupskfold = groups.values

parameters = {
    'n_estimators' :[10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000],
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(2, 40, 2),
    'min_samples_split': [2, 10, 50, 70,100,120,150,180, 200, 250,400,600,1000, 1300, 2000],
    'min_samples_leaf': [5, 10, 15, 20, 25, 30, 35, 40, 45],
    'max_features': list(range(1,X.shape[1])),
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'class_weight': [{0:4,1:6},{0:1,1:10},{0:1,1:50},{0:1,1:70}]
}

best_scores = []
best_parameters = []
full_scores = []

folds =[10]


columns_sel = ['param_n_estimators', 'param_max_features', 'param_max_depth',
               'param_criterion','param_bootstrap', 'params', 'mean_test_acc', 'mean_train_acc', 'mean_test_AUC', 'mean_train_AUC',
               'mean_test_prec', 'mean_train_prec', 'mean_test_rec', 'mean_train_rec', 'rank_test_f_score', 'mean_train_f_score','folds']

results = pd.DataFrame(columns=columns_sel)

for i in folds:
    print("\ncv = ", i)
    best_params, best_score, full_scores = hyperparameter_tune(model, parameters, i, X, Y,groupskfold)

    df_results = pd.DataFrame.from_dict(full_scores)
    df_short = df_results.filter(regex="mean|std|params")
   # df_results.to_csv('/home/sgirtsou/Documents/GridSearchCV/ExtraTrees/split'+str(i)+'_corine_l2_groups.csv')
    df_short.to_csv('/home/sgirtsou/Documents/GridSearchCV/ExtraTrees/extratrees_random_search_old_dataset_new_feat.csv')
    '''
    df1 = df_results[columns_sel]
    df_no_split_cols = [c for c in df_results.columns if 'split' not in c]

    df_results.to_csv('rfresults.csv')
    df_results[df_no_split_cols].to_csv('rfresults_nosplit.csv')

    results = pd.concat([results, df1])

    best_scores.append(best_score)
    best_parameters.append(best_params)
    '''