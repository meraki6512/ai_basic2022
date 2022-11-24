# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:47:48 2022

@author: 82103
"""

## 최근접 알고리즘 관련 테스트
# - 분류 모델

import pandas as pd

# 빈 데이터 프레임 생성
X = pd.DataFrame()
print(X)

X['rate'] = [0.3, 0.8, 0.999]
print(X)

X['price'] = [10000, 5000, 9500]
print(X)

# 종속 변수 생성
y = pd.Series([0, 1, 0])





# 스케일 전처리 과정 수행
# - price가 rate보다 압도적으로 스케일이 크다...
# - price 컬럼의 값을 rate 컬럼의 값과 동일한 범위를 가지도록 데이터를 수정.
#   (데이터의 값은 수정되지만 원본 값에서 가지는 상대적인 크기는 유지)
# - sklearn.preprocessing
# - MinMaxScaler, StandardScaler, RobustScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# MinMaxScaler 전처리 과정 수행
# 1. 각 컬럼별 최대 / 최소값 추출
# 2. 각 컬럼별 아래의 연산을 수행하여 값을 대체
# - (원본값-최소값)/(최대값-최소값)
# - 결과적으로 모든 컬럼의 값은 ... 최대값=1, 최소값=0으로 변환이 된다. 

scaler.fit(X)

# 스케일 처리 수행 코드 (실제로 변환되는 코드)
X = scaler.transform(X)
print(X)






from sklearn.neighbors import KNeighborsClassifier

# 가장 인접한 1개 데이터를 기준으로 판단
model = KNeighborsClassifier(n_neighbors=1)

model.fit(X, y)

# 예측할 데이터
X_new = [[0.81, 7000]]
X_new = scaler.transform(X_new)

# 예측 수행
pred = model.predict(X_new)
print(pred)

# 학습에 사용된 X 데이터
#     rate  price
# 0  0.300  10000
# 1  0.800   5000
# 2  0.999   9500

# 학습에 사용된 y 데이터
# [0, 1, 0]


# <최근접 이웃 알고리즘의 학습 및 예측 방법>
# - 학습: fit 메소드에 입력된 데이터를 단순 저장.
# - 예측: fit 메소드에 의해서 저장된 데이터와
#              예측하고자 하는 신규 데이터 와의 유클리드 거리를 계산.
#                                           (다른 방식 사용 ... ski-kit 페이지 확인)
#              -> 가장 인접한 n_neighbors 개수를 사용해 이웃 추출.
#              -> 추출된 이웃의 y값을 사용해 "voting(다수결)"의 과정을 수행.

# 유클리드 거리
# : 동일 특성간의 차이 계산
# ex) (rate-new_rate)**2 + (price-new_price)**2 












