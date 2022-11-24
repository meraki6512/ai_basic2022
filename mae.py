# -*- coding: utf-8 -*-

# 1. 데이터
import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

fname = "./house_prices.csv"
data = pd.read_csv(fname, header = "infer", sep =",")

print(data.info())
print(data.describe())

print(data['Id'].value_counts())
data = data.iloc[:, 1:]




# 2. 전처리 : 결측치처리 + 인코딩 + 스케일링
#결측 데이터 삭제
#not_nan_series = data.isnull().sum()
#not_nan_series = not_nan_series[not_nan_series == 0]
#not_nan_columns = not_nan_series.index.tolist()
#data = data[not_nan_columns]

X = data.iloc[:, :-1]
y = data.SalePrice

num_col = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
obj_col = [cname for cname in X.columns if X[cname].dtype == 'object']


import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder #이경우는??
from sklearn.preprocessing import OneHotEncoder


num_imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean')

obj_imputer = SimpleImputer(
    missing_values=np.nan, 
    strategy='most_frequent')

scaler = MinMaxScaler()
#encoder = LabelEncoder()    
encoder = OneHotEncoder(sparse = False,
                       handle_unknown='ignore')


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_pipe = Pipeline([('num_imputer', num_imputer),
                 ('scaler', scaler)])
obj_pipe = Pipeline([('obj_imputer', obj_imputer),
                     ('encoder', encoder)])

ct = ColumnTransformer([('num_pipe', num_pipe, num_col),
                        ('obj_pipe', obj_pipe, obj_col)])

ct.fit(X)
X = ct.transform(X)

print(X.shape)




# 3. 학습과 평가
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3,
                                                    random_state = 11)


from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

model_knn = KNeighborsRegressor(n_neighbors=5,
                                n_jobs=-1).fit(X_train, y_train)
model_rf = RandomForestRegressor(n_estimators=100,
                                 max_depth=None,
                                 n_jobs=-1,
                                 random_state=11).fit(X_train, y_train)
model_gb = GradientBoostingRegressor(n_estimators=200,
                                     max_depth=1,
                                     subsample=0.3,
                                     random_state=11).fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

score_knn = mean_absolute_error(y_test, model_knn.predict(X_test))
score_rf = mean_absolute_error(y_test, model_rf.predict(X_test))
score_gb = mean_absolute_error(y_test, model_gb.predict(X_test))

print(f'score_knn : {score_knn}')
print(f'score_rf : {score_rf}')
print(f'score_gb : {score_gb}') 


best_model = model_rf
print(data.SalePrice.mean())
print(data.SalePrice.std())
score_r2 = best_model.score(X_test, y_test)
score_mae = mean_absolute_error(y_test, best_model.predict(X_test))
print(f'score_r2 : {score_r2}')
print(f'score_mae: {score_mae}')




