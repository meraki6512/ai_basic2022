# -*- coding: utf-8 -*-

# 1. 데이터 불러오기
import numpy as np
import pandas as pd

X = pd.DataFrame()

print(X)
print(X.info())

X['gender'] = ['F', 'M', 'F', 'F', None]
X['age'] = [15, None, 25, 37, 55]

print(X)
print(X.info())
print(X.isnull().sum())

# 2. 전처리 ( 결측치처리 / 문자열인코딩 / 수치형스케일링 )
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

encoder = OneHotEncoder(sparse = False, 
                        handle_unknown = 'ignore')
scaler = MinMaxScaler()

imputer_num = SimpleImputer(
    missing_values=np.nan,
    strategy='mean'
    )

imputer_obj = SimpleImputer(
    missing_values=None,
    strategy='most_frequent'
    )

from sklearn.pipeline import Pipeline

num_pipe = Pipeline([('imputer_num', imputer_num),
                     ('scaler', scaler)])
obj_pipe = Pipeline([('imputer_obj', imputer_obj),
                     ('encoder', encoder)])


from sklearn.compose import ColumnTransformer

obj_columns = ['gender']
num_columns = ['age']

#ct = ColumnTransformer([('imputer_num', imputer_num, num_columns),
#                        ('scaler', scaler, num_columns),
#                        ('imputer_obj', imputer_obj, obj_columns),
#                        ('encoder', encoder, obj_columns)])

ct = ColumnTransformer([('num_pipe', num_pipe, num_columns),
                        ('obj_pipe', obj_pipe, obj_columns)])

ct.fit(X)
print(X)
print(ct.transform(X))






























