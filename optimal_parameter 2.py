# -*- coding: utf-8 -*-

# 1. 데이터 불러오기, 체크, 전처리-스케일링
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# 2. 최적 하이퍼파라미터 찾기 - " 조건부 매개변수 "
# 모델: LogisticRegression
# 파라미터: penalty, c, solver, max_iter, class_weight, random_state, n_jobs ...

# param_grid = {'C':[1., 0.1, 0.01, 10, 100],
#               'penalty':['l2', 'elasticnet'],
#               'solver':['lbfgs', 'liblinear', 'sag', 'saga']}
param_grid = [{'C':[1., 0.1, 0.01, 10, 100],
              'penalty':['l1'],
              'solver':['liblinear', 'saga']},
              {'C':[1., 0.1, 0.01, 10, 100],
               'penalty':['l2'],
               'solver':['lbfgs', 'sag', 'saga']},
              {'C':[1., 0.1, 0.01, 10, 100],
               'penalty':['elasticnet'],
               'solver':['saga']}]


from sklearn.model_selection import GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=1)
estimator = LogisticRegression(max_iter=10000)

grid_model = GridSearchCV(estimator=estimator,
                          param_grid=param_grid,
                          cv=cv,
                          n_jobs=-1).fit(X_train, y_train)

grid_model.best_estimator_
grid_model.best_params_
grid_model.best_score_

grid_model.score(X_train, y_train)
grid_model.score(X_test, y_test)
































