import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import xgboost as xgb

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

data = pd.read_csv('/home/sgirtsou/Documents/ML-dataset_newLU/dataset_dummies.csv')

X = data[['max_temp', 'min_temp', 'mean_temp',
       'res_max', 'dom_vel', 'rain_7days', 'Near_dist', 'DEM', 'Slope',
       'Curvature', 'Aspect', 'ndvi_new', 'evi', 'dir_max_1',
       'dir_max_2', 'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6',
       'dir_max_7', 'dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3',
       'dom_dir_4', 'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8',
       'fclass_bridleway', 'fclass_footway', 'fclass_living_street',
       'fclass_motorway', 'fclass_path', 'fclass_pedestrian', 'fclass_primary',
       'fclass_residential', 'fclass_secondary', 'fclass_service',
       'fclass_steps', 'fclass_tertiary', 'fclass_track',
       'fclass_track_grade1', 'fclass_track_grade2', 'fclass_track_grade3',
       'fclass_track_grade4', 'fclass_track_grade5', 'fclass_trunk',
       'fclass_unclassified', 'fclass_unknown', 'Corine_111', 'Corine_112',
       'Corine_121', 'Corine_122', 'Corine_123', 'Corine_131', 'Corine_132',
       'Corine_133', 'Corine_142', 'Corine_211', 'Corine_212', 'Corine_213',
       'Corine_221', 'Corine_222', 'Corine_223', 'Corine_231', 'Corine_241',
       'Corine_242', 'Corine_243', 'Corine_311', 'Corine_312', 'Corine_313',
       'Corine_321', 'Corine_322', 'Corine_323', 'Corine_324', 'Corine_331',
       'Corine_332', 'Corine_333', 'Corine_334', 'Corine_411', 'Corine_421',
       'Corine_511', 'Corine_512', 'Corine_523']]
y = data['fire']
data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

'''
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
'''

accuracy = accuracy_score(y_test, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
i = 1