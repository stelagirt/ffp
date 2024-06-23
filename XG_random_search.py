import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
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

    optimal_model = RandomizedSearchCV(base_model, parameters,scoring=metrics, n_iter=500, cv=k, verbose=3,refit='rec_1', return_train_score=True)
    optimal_model.fit(X, Y,groups)

    return optimal_model.best_params_, optimal_model.best_score_, optimal_model.cv_results_


#data = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_dummies.csv')
#data = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_corine_level2_onehotenc.csv')
#data = pd.read_csv('/home/sgirtsou/Documents/dataset_120/dataset_1_10_corine_level2_onehotenc.csv')
data = pd.read_csv('/home/sgirtsou/Documents/datasets/training_dataset_dew_lst_dummies.csv')

X = data[['max_temp', 'min_temp', 'mean_temp',
       'res_max', 'dom_vel', 'rain_7days', 'dem', 'slope', 'curvature',
       'aspect', 'ndvi_new', 'evi', 'max_dew_temp', 'mean_dew_temp',
       'min_dew_temp', 'lst_day', 'lst_night', 'dir_max_1.0',
       'dir_max_2.0', 'dir_max_3.0', 'dir_max_4.0', 'dir_max_5.0',
       'dir_max_6.0', 'dir_max_7.0', 'dir_max_8.0', 'dom_dir_1.0',
       'dom_dir_2.0', 'dom_dir_3.0', 'dom_dir_4.0', 'dom_dir_5.0',
       'dom_dir_6.0', 'dom_dir_7.0', 'dom_dir_8.0', 'corine_111.0',
       'corine_112.0', 'corine_121.0', 'corine_122.0', 'corine_123.0',
       'corine_131.0', 'corine_132.0', 'corine_133.0', 'corine_142.0',
       'corine_211.0', 'corine_212.0', 'corine_213.0', 'corine_221.0',
       'corine_222.0', 'corine_223.0', 'corine_231.0', 'corine_241.0',
       'corine_242.0', 'corine_243.0', 'corine_311.0', 'corine_312.0',
       'corine_313.0', 'corine_321.0', 'corine_322.0', 'corine_323.0',
       'corine_324.0', 'corine_331.0', 'corine_332.0', 'corine_333.0',
       'corine_334.0', 'corine_411.0', 'corine_421.0', 'corine_511.0',
       'corine_512.0', 'corine_523.0']]
Y = data['fire']

model = XGBClassifier(n_jobs =4)

groups = data['firedate']
groupskfold = groups.values

parameters = {
    'max_depth' : range(2, 100, 2),
    'n_estimators' :[10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000],
    #'scale_pos_weight': range(1, 400, 50),
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'alpha' : [0, 1, 10, 20, 40, 60, 80, 100],
    'gamma' : [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'lambda' : range(1, 22, 1),
   # 'scale_pos_weight': [6,7,8,9,15,50,70]
    'scale_pos_weight': [9,15,50,70,100,200,500]
}

best_scores = []
best_parameters = []
full_scores = []

folds =[10]


columns_sel = ['param_n_estimators', 'param_max_features', 'param_max_depth',
               'param_criterion','param_bootstrap', 'params', 'mean_test_acc', 'mean_train_acc', 'mean_test_AUC', 'mean_train_AUC',
               'mean_test_prec', 'mean_train_prec', 'mean_test_rec', 'mean_train_rec', 'rank_test_f_score','mean_train_f_score','folds']

results = pd.DataFrame(columns=columns_sel)

for i in folds:
    print("\ncv = ", i)
    best_params, best_score, full_scores = hyperparameter_tune(model, parameters, i, X, Y,groupskfold)

    df_results = pd.DataFrame.from_dict(full_scores)
    df_results['folds'] = int(i)
    #df_results.to_csv('/home/sgirtsou/Documents/GridSearchCV/XG/split'+str(i)+'_withauc.csv')
    df_short = df_results.filter(regex="mean|std|params")
    df_short.to_csv('/home/sgirtsou/Documents/GridSearchCV/LB/LBcv_dataset_balanced_lst_dew_noshufflestrictcriterion_500.csv')

    '''
    df1 = df_results[columns_sel]
    df_no_split_cols = [c for c in df_results.columns if 'split' not in c]

    df_results.to_csv('rfresults.csv')
    df_results[df_no_split_cols].to_csv('rfresults_nosplit.csv')

    results = pd.concat([results, df1])

    best_scores.append(best_score)
    best_parameters.append(best_params)
    '''
