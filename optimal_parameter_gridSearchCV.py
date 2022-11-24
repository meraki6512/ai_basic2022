# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)  # pd.DataFrame으로 만들 경우 ???
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    stratify=y, 
                                                    random_state=1)

# 최적의 하이퍼파라미터 찾기
# GridSearchCV # (번외로 RandomizedSearchCV도 있음.)
from sklearn.model_selection import GridSearchCV

param_grid = {'learning_rate' : [0.1, 0.2, 0.3, 1., 0.01],
              'max_depth' : [1, 2, 3],
              'n_estimators' : [100, 200, 300, 10, 50]}
cv = KFold(n_splits=5, shuffle=True, random_state=1)
base_model = GradientBoostingClassifier(random_state=1)

grid_model = GridSearchCV(estimator=base_model, 
                          param_grid=param_grid,
                          cv=cv,
                          n_jobs=-1,
                          verbose=3,
                          scoring='accuracy')
grid_model.fit(X_train, y_train)

print(f'best_score : {grid_model.best_score_}')
print(f'best_param : {grid_model.best_params_}')
print(f'best_model : {grid_model.best_estimator_}')



# best 모델 생성, 학습, 평가
print(f'Score(TRAIN) : {grid_model.score(X_train, y_train)}')
print(f'Score(TEST) : {grid_model.score(X_test, y_test)}')


