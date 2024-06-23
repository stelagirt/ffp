import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import feature_selection
import time
import warnings
import csv


def normalized_values(y, dfmax, dfmin, dfmean, dfstd, t=None):
    if not t:
        a = (y - dfmin) / (dfmax - dfmin)
        return (a)
    elif t == 'std':
        a = (y - dfmean) / dfstd
        return (a)
    elif t == 'no':
        return y


def normalize_dataset(X_unnorm_int, norm_type=None):
    X = pd.DataFrame()
    for c in X_unnorm_int.columns:
        print(c)
        dfmax = X_unnorm_int[c].max()
        dfmin = X_unnorm_int[c].min()
        dfmean = X_unnorm_int[c].mean()
        dfstd = X_unnorm_int[c].std()
        X[c] = X_unnorm_int.apply(lambda x: normalized_values(x[c], dfmax, dfmin, dfmean, dfstd, norm_type), axis=1)
    return X

def cmvals(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    return tn, fp, fn, tp

def specificity(y_true, y_pred):
    tn, fp, fn, tp = cmvals(y_true, y_pred)
    return 1.0*tn/(tn+fp)*1.0

def NPV(y_true, y_pred):
    tn, fp, fn, tp = cmvals(y_true, y_pred)
    return 1.0*tn/(tn+fn)*1.0

def hyperparameter_tune(base_model, parameters, kfold, X, y):
    start_time = time.time()

    # Arrange data into folds with approx equal proportion of classes within each fold
    #k = StratifiedKFold(n_splits=kfold, shuffle=True)
    k = KFold(n_splits=kfold, shuffle=True)
    #k = GroupKFold(n_splits=kfold)

    scoring_st = {'acc': 'accuracy',
                  'AUC': 'roc_auc',
                  'prec': 'precision',
                  'rec': 'recall',
                  'f_score': 'f1'
    }
    optimal_model = RandomizedSearchCV(base_model,
                                       param_distributions=parameters,
                                       n_iter=15,
                                       cv=k,
                                       scoring = scoring_st,
                                       n_jobs=6,
                                       refit='rec',
                                       return_train_score=True)
                                       #random_state=SEED)

    optimal_model.fit(X, y)

    stop_time = time.time()



    #scores = cross_validate(optimal_model, X, y, cv=k, scoring= scoring_st, return_train_score=True, return_estimator=True)

    print("Elapsed Time:", time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time)))
    print("====================")
    #print("Cross Val Mean: {:.3f}, Cross Val Stdev: {:.3f}".format(scores.mean(), scores.std()))
    print("Best Score: {:.3f}".format(optimal_model.best_score_))
    print("Best Parameters: {}".format(optimal_model.best_params_))


    return optimal_model.best_params_, optimal_model.best_score_, optimal_model.cv_results_

#df = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/training_dataset.csv')
#df = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_ndvi_lu.csv')
#df = pd.read_csv('/home/sgirtsou/PycharmProjects/ML/ML_fires_al/dataset_dummies.csv')
#df = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_corine_level2_onehotenc.csv')
df = pd.read_csv('/home/sgirtsou/Documents/perifereia/13082013_final.csv')

df_part = df[['id','dem_tif', 'lu_tif', 'aspect_tif', 'slope_tif', 'fire', 'firedate', 'x', 'y', 'max_temp',
       'min_temp', 'mean_temp', 'res_max', 'dir_max', 'dom_vel', 'dom_dir','ndvi']].copy()

X_unnorm, y_int = df_part[['dem_tif', 'lu_tif', 'aspect_tif', 'slope_tif','max_temp',
       'min_temp', 'mean_temp','res_max', 'dir_max', 'dom_vel', 'dom_dir','ndvi']], df_part['fire']


print(df.columns)

#groups = df['firedate']

#X = normalize_dataset(X_unnorm, 'std')
y = y_int

X_ = X_unnorm.values
#X_ = X.values
y_ = y.values
#groupskfold = groups.values

rf = RandomForestClassifier(n_jobs=-1)
depth = [10, 20, 100, 200, 400,500, 700, 1000, 1200,2000, None]
n_estimators = [50, 100, 120, 150,170,200, 250, 350, 500, 750, 1000,1400, 1500]
min_samples_split = [2, 10, 50, 70,100,120,150,180, 200, 250,400,600,1000, 1300, 2000]
min_samples_leaf = [1, 10,30,40,50,100,120,150] #with numbers
max_features = list(range(1,X_.shape[1]))
bootstrap = [True, False]
criterion = ["gini", "entropy"]
class_weights = [{0:1,1:5},{0:1,1:10},{0:1,1:25},{0:1,1:50}, {0:1,1:100}, {0:1,1:200}]


lots_of_parameters = {
    "max_depth": depth, #depth of each tree
    "n_estimators": n_estimators, #trees of the forest
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "criterion": criterion,
    "max_features": max_features,
    "bootstrap": bootstrap,
    "class_weight": class_weights
}


best_scores = []
best_parameters = []
full_scores = []
folds = [10]#range(2, 8)

columns_sel = ['mean_test_acc','std_test_acc', 'mean_train_acc', 'std_train_acc','mean_test_AUC','std_test_AUC', 'mean_train_AUC', 'std_train_AUC',
               'mean_test_prec','std_test_prec','mean_train_prec','std_train_prec', 'mean_test_rec','std_test_rec',
               'mean_train_rec','std_train_rec', 'mean_test_f_score','std_test_f_score', 'mean_train_f_score','std_train_f_score',
                'params','folds']

results = pd.DataFrame(columns=columns_sel)

for i in folds:
    print("\ncv = ", i)
    start = time.time()
    best_params, best_score, full_scores = hyperparameter_tune(rf, lots_of_parameters, i, X_, y_)

    df_results = pd.DataFrame.from_dict(full_scores)
    df_results['folds'] = int(i)
    #df_results.to_csv('/home/sgirtsou/Documents/GridSearchCV/RF/RFcv_25kbalanced_noshufflestrictcriterion.csv')

    df1 = df_results[columns_sel]
    #df_no_split_cols = [c for c in df_results.columns if 'split' not in c]
    #df1.to_csv('/home/sgirtsou/Documents/GridSearchCV/RF/RFcv_25kbalanced_noshufflestrictcriterion_sh.csv')

    results = pd.concat([results, df1])

    best_scores.append(best_score)
    best_parameters.append(best_params)
    end = time.time()
    dur = end-start
    print(dur)
#results.to_csv('/home/sgirtsou/Documents/GridSearchCV/random_search2/rscv_total.csv')
i = 1