# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer # 문자열 인코딩 전처리를 추가한다면 ??
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV


# 1. 데이터 체크
data = fetch_california_housing()

X= pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(X.info())
print(X.describe())
print(y.head())
print(y.describe())

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=1)

# 3. 파이프라이닝 = 전처리(column tranformer) + 모델 
# - 전처리
    # 결측치 없음
    # 인코딩 x : 문자열 없음
    # 스케일링 o ( + 이상치 제거도 가능하면 ㄱ)

s_mm = MinMaxScaler()
s_std = StandardScaler() 
s_rb = RobustScaler()

mm_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms'] # X.iloc[:,0:4].columns
std_cols = ['AveOccup', 'Latitude', 'Longitude']
rb_cols = ['Population']

ct = ColumnTransformer([('s_mm', s_mm, mm_cols),
                        ('s_std', s_std, std_cols),
                        ('s_rb', s_rb, rb_cols)],
                       n_jobs=-1)

# - 모델
    # RandomForestRegressor # 성능 올리려면 ... 학습 약하게 제약 풀어줌    
model = RandomForestRegressor(n_jobs=-1, random_state=1)

# - 파이프라이닝 -> GridSearchCV
pipe = Pipeline([('ct', ct),
                 ('model', model)])
 
cv = KFold(n_splits=15, shuffle=True, random_state=1)
param_grid = {'model__n_estimators':[100, 50, 10, 200],
              'model__max_depth':[None, 7, 10]}
grid = GridSearchCV(pipe, 
                    param_grid=param_grid,
                    cv=cv,
                    n_jobs=-1,
                    verbose=3,
                    scoring='r2').fit(X_train, y_train)

print(f'best_score:{grid.best_score_}')
grid.best_estimator_
grid.best_params_

# 4. 성능 평가
print(f'score(train) : {grid.score(X_train, y_train)}')
print(f'score(test) : {grid.score(X_test, y_test)}')




# 편하고 간단하지만 실행속도는 느릴 수 있다!










