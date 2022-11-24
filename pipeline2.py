# -*- coding: utf-8 -*-


# 1. 데이터 불러오기, 체크, 전처리-스케일링
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=1)
import pandas as pd
X_df = pd.DataFrame(X)
print(X_df.describe(include='all'))


# 2. pipeline -> scaler + gridsearchcv

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()#.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


from sklearn.pipeline import Pipeline
base_model = LogisticRegression(n_jobs=-1, random_state=1)
pipe = Pipeline([('s_scaler', scaler),
                 ('base_model', base_model)])
print(pipe.fit(X_train, y_train))
print(pipe.score(X_train, y_train))

param_grid = [{'base_model__C':[1., 0.1, 0.01, 10, 100],
               'base_model__penalty':['l2'],
               'base_model__solver':['lbfgs'],
               'base_model__class_weight':['balanced', {0:0.9, 1:0.1}]},
              {'base_model__C':[1., 0.1, 0.01, 10, 100],
               'base_model__penalty':['elasticnet'],
               'base_model__solver':['saga'],
               'base_model__class_weight':['balanced', {0:0.9, 1:0.1}]}]


cv = KFold(n_splits=5, shuffle=True, random_state=1)
estimator = LogisticRegression(n_jobs=-1, random_state=1, max_iter=10000)

grid_model = GridSearchCV(pipe,
                          param_grid=param_grid,
                          cv=cv,
                          n_jobs=-1,
                          scoring='recall').fit(X_train, y_train)



grid_model.best_estimator_
grid_model.best_params_
grid_model.best_score_

grid_model.score(X_train, y_train)
grid_model.score(X_test, y_test)

from sklearn.metrics import classification_report
pred_train = grid_model.predict(X_train)
print(classification_report(y_train, pred_train))
pred_test = grid_model.predict(X_test)
print(classification_report(y_test, pred_test))





























