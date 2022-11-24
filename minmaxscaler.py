# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target)

print(X.info())
print(X.describe())

# [데이터 전처리 방법 리뷰]
# 문자열 
    # -> 결측 데이터
    # -> 라벨 인코딩
    # -> 원핫 인코딩
# 수치형 
    # -> 결측 데이터
    # -> 스케일링
    # -> (이상치 제거, 대체)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

num_columns = X.columns

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('scaler', scaler, num_columns)])

ct.fit(X)

print(X.head())
print(ct.transform(X)[:5])

























