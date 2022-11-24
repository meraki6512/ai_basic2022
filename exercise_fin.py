# -*- coding: utf-8 -*-

# 파일의 종류: 
#     분류형 / 회귀형
#     결측치 존재 (처리)
#     문자열 존재 (인코딩)
#     수치형 오차 (스케일링)
    
# 스케일링 전처리 종류:
#     MinMaxScaler
#     StandardScaler
#     RobustScaler
    
# 구현 종류:
#     SimpleImpute
#     ColumnTransformer
#     Pipeline (다양한 전처리들 + 모델)
#     KFold, cross_val_score / GridSearchCV, 조건부 매개변수
    
# 모델 종류:
#     KNeighbors (Classifier, Regressor)
#     LinearRegression
#     LogisticRegressor
#     DecisionTree (Classifier, Regressor)
#     RandomForest (Classifier, Regressor)
#     GradientBoosting (Classifier, Regressor)
        
# 스태킹 구현

# ################################################################

# 혹시 몰라 체크:
#     StratifiedKFold
#     RandomSearch
#     LabelEncoding

# ################################################################

# 1) 데이터: ./data/house_prices

import numpy as np
import pandas as pd
pd.options.display.max_columns = 100

data = pd.read_csv('./data/house_prices.csv', header='infer', sep=',')
print(data.info())
print(data.describe())
print(data.Id.value_counts())

# Id 성격 제거
data = data.iloc[:,1:]
X = data.iloc[:,:-1]
y = data.SalePrice
print(X.info())
print(X.describe())
print(X.head())
print(y.describe())
print(y.head())

# 결측치? 처리 (simle impute)
# 문자열? 원핫인코딩 
# 수치형? MinMax, Standard, Robust 스케일링
# 이상치 제거
# 회귀형 모델: gradientboosting / randomforest 앙상블 linearregression, decisiontree, kneighbors 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, GridSearchCV 

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=1)

# 2) 결측치, 문자열, 수치형 전처리

num_col = [cname for cname in X.columns if (X[cname].dtype in ['float64', 'int64'])]
obj_col = [cname for cname in X.columns if (X[cname].dtype == 'object')]

num_impute = SimpleImputer(missing_values=np.nan, 
                           strategy='mean')
obj_impute = SimpleImputer(missing_values=np.nan,
                           strategy='most_frequent')

print(X[num_col].describe())

mm_col = ['MSSubClass', 'OverallQual', 'OverallCond', 
          'YearBuilt', 'YearRemodAdd', 'BsmtFullBath',
          'BsmtHalfBath', 'FullBath', 'HalfBath',
          'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
          'Fireplaces', 'GarageCars','MoSold', 'YrSold']
s_col = ['LotFrontage', 'LowQualFinSF']
r_col = ['LotArea', 'MasVnrArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
         '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageYrBlt',
         'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
         '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

mm_scaler = MinMaxScaler()
s_scaler = StandardScaler()
r_scaler = RobustScaler()

print(X[obj_col].describe())
print(X[obj_col].head())
encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')


mm_pipe = Pipeline([('num_impute', num_impute),
                     ('mm_scaler', mm_scaler)])
s_pipe = Pipeline([('num_impute', num_impute),
                     ('s_scaler', s_scaler)])
r_pipe = Pipeline([('num_impute', num_impute),
                     ('r_scaler', r_scaler)])
                   
ct = ColumnTransformer([('mm_pipe', mm_pipe, mm_col),
                        ('s_pipe', s_pipe, s_col),
                        ('r_pipe', r_pipe, r_col),
                        ('encoder', encoder, obj_col)],
                       n_jobs=-1)

# 최적 하이퍼파라미터 + 모델 생성
model = GradientBoostingRegressor(random_state=1)

pipe = Pipeline([('ct', ct),
                 ('model', model)])

cv = KFold(n_splits=5, shuffle=True, random_state=1)
param_grid = {'model__learning_rate': [0.1, 1, 5, 10, 100],
              'model__n_estimators': [10, 50, 100, 200, 300],
              #'model__subsample': [0.1, 0.2, 0.3],
              'model__max_depth': [1,2,3]}

grid = GridSearchCV(pipe, 
                    cv=cv,
                    param_grid=param_grid,
                    scoring='r2',
                    n_jobs=-1).fit(X_train, y_train)

print(grid.best_estimator_)
print(grid.best_params_)
print(grid.best_score_)

train_score = grid.score(X_train, y_train)
test_score = grid.score(X_test, y_test)
print(f'score(Train) : {train_score}')
print(f'score(TEST) : {test_score}')




































    