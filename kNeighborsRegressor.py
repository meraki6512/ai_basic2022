# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:04:30 2022

@author: 82103
"""

## 최근접 알고리즘 관련 테스트
# - 회귀 모델
# - 한계점(과 그 대안의 예시)

import numpy as np

# 설명변수 생성
# 1차원 배열 생성
# - arange
X = np.arange(1,11) #1부터 11전까지 증감률:1
print(X)
# 1차원 배열을 2차원 배열로 수정
X = X.reshape(-1, 1) #(row, col) #-1:미정
print(X)

# 종속변수 생성 - 연속된 수치형
y = np.arange(10, 101, 10) #10부터 101전까지 증감률:10
print(y)


# KNeighborsRegressor: 회귀 예측을 수행하는 클래스
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=2)
model.fit(X, y)
# 예측 수행
X_new = [[3.7]]
pred = model.predict(X_new)
print(pred)

# <최근접 이웃 알고리즘의 학습 및 예측 방법>
# - 학습: fit 메소드에 입력된 데이터를 단순 저장.
# - 예측: fit 메소드에 의해서 저장된 데이터와
#              예측하고자 하는 신규 데이터 와의 유클리드 거리를 계산.
#              -> 가장 인접한 n_neighbors 개수를 사용해 이웃 추출.
#              -> 추출된 이웃의 y값을 사용해 "평균 값"을 반환.






# 학습에 사용된 X 데이터
# [[ 1]
#  [ 2]
#  [ 3]
#  [ 4]
#  [ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]

# X의 범위를 넘어가는 값이 주어짐.
X_new = [[57.7]]
pred = model.predict(X_new)
print(pred) # 결과 [95.]

X_new = [[10000000.7]]
pred = model.predict(X_new)
print(pred) # 결과 [95.]

# 최근접 이웃 알고리즘의 한계점(단점)
# - fit 메소드에 입력된 X 데이터의 범위를 벗어나면,
#   양 끝단의 값으로만 예측을 수행한다.
#   (즉, 학습 시에 저장된 값으로만 예측이 가능하다.)







# 대체 방안 중 한 가지 예시
# linear_model - LinearRegression
# 선형 방정식을 기반으로 회귀 분석을 수행하는 머신러닝 클래스
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

X_new = [[3.7]]
pred = model.predict(X_new)
print(pred)

X_new = [[57.7]]
pred = model.predict(X_new)
print(pred)

X_new = [[-10.7]]
pred = model.predict(X_new)
print(pred)










