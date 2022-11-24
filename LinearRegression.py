# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:09:21 2022

@author: 82103
"""

# 회귀 분석
#   - 머신러닝이 예측해야하는 정답의 데이터가 연속된 수치형인 경우를 의미함.
#   (분류 분석 - 정답 데이터는 범주형 [남/여, 유/무, ..])
#   - 선형 방정식을 활용한 머신러닝 실습

import pandas as pd

#load_ : 연습용 데이터 셋
#fetch_: 실 데이터 셋 (데이터 수가 상대적으로 많음)
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

# 설명변수 (2차원 데이터): 집값 데이터를 예측하기 위한 피처 데이터를 저장하는 변수 
X = pd.DataFrame(data.data, columns = data.feature_names)
# 종속변수 (1차원 데이터, 설명변수의 row수와 동일한 크기): 설명변수를 사용하여 예측하기 위한 변수
y = pd.Series(data.target)

# X의 EDA
print(X.info())    
    
# 결측 데이터 확인
print(X.isnull())    
print(X.isnull().sum()) #count    


pd.options.display.max_columns = 100
print(X.describe(include='all'))


# X 데이터를 구성하는 각 피처(특성)들의 스케일(범위) 반드시 체크
#   - Populaion 컬럼에서 스케일 차이 발생함을 확인할 수 있다.
# 스케일 전처리 방법
#       - 정규화, 일반화
#       - StandartScaler, MinMaxScaler

# 전처리 체크
# - 각 컬럼(피처, 특성)들에 대해서 산포도, 비율 등을 시각화하는 과정


# 종속변수 확인
# - 연속형 수치 데이터임을 확인할 수 있다.
# -> 회귀 분석을 위한 데이터 셋
print(y.head())
print(y.tail())

# 회귀 분석의 경우, 중복되는 데이터가 흔치 않으므로
# 분류 분석에서와 같이 value_counts를 사용해 값의 개수를 확인하는 과정은 
# 일반적으로 생략한다.
# print(y.value_counts())


# 데이터 분할
# stratify(층화추출방법)를 사용하지 않는 이유:
#   회귀 분석을 위한 데이터 셋의 경우, (정답 데이터가 연속형)
#   y 데이터 내부의 값의 분포 비율을 유지할 필요가 (일반적으로) 없음
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# shape 속성: 데이터의 차원 정보를 반환한다.
print(X_train.shape, X_test.shape)
# len 함수 사용도 가능하다.
print(len(X_train), len(X_test))
print(len(y_train), len(y_test))

# 선형 방정식을 기반으로 회귀 예측을 수행할 수 있는 클래스
#    (y = x1*w1 + ... xN*wN + b) 
# LinearRegression 클래스의 학습은
# X 데이터를 구성하는 각 컬럼(피처, 특성)별 
# 최적화된 가중치와 절편의 값을 계산하는 과정을 수행
from sklearn.linear_model import LinearRegression

# 머신 러닝의 객체 생성
# - n_jobs: 병렬 처리를 위함 (-1: 현재 사용가능한 코어)
model = LinearRegression(n_jobs=-1)

# 학습 (fit 메소드)
model.fit(X_train, y_train)


# 평가 (score 메소드)
# - 분류를 위한 클래스: 정확도(Accuracy): 전체 데이터 중 정답으로 맞춘 비율
# - 회귀를 위한 클래스: 결정계수(R2 Score): n<=-1의 범위를 가지는 평가값

# 결정계수(R2) 계산 공식
#   1 - 
#       (y - y_train)의제곱값합계 /
#       (y - y의 평균값)의제곱값합계
# R2 값 == 0 :
#       머신러닝 모델이 예측하는 값이 실제 정답의 평균값과 일치하는 경우
#       (학습 효과가 낮음)
# R2 값 == 1 :
#       머신러닝 모델이 예측하는 값이 실제 정답과 완벽하게 일치하는 경우   
#       (학습 효과가 좋음)
# R2 값 < 0 :
#       머신러닝 모델이 예측하는 값이 실제 정답의 평균값도 예측하지 못하는 경우
#       (학습이 부족함)

score = model.score(X_train, y_train)
print(f'Train: {score}')


# 테스트
score = model.score(X_test, y_test)
print(f'Test: {score}')


# 예측
# 테스트 데이터의 가장 앞 데이터를 사용해 예측 수행
pred = model.predict(X_test.iloc[:1])
print(pred)


# 선형 방정식을 기반으로 회귀 예측을 수행할 수 있는 클래스
#    (y = x1*w1 + ... xN*wN + b) 
# 머신러닝 모델이 학습한 기울기(가중치), 절편을 확인
# - 기울기(가중치)
print(model.coef_)
# - 절편(bias)
print(model.intercept_)


X_test.iloc[:1]
pred = X_test.iat[0,0] * model.coef_[0] + \
    X_test.iat[0,1] * model.coef_[1] + \
        X_test.iat[0,2] * model.coef_[2] + \
            X_test.iat[0,3] * model.coef_[3] + \
                X_test.iat[0,4] * model.coef_[4] + \
                    X_test.iat[0,5] * model.coef_[5] + \
                        X_test.iat[0,6] * model.coef_[6] + \
                            X_test.iat[0,7] * model.coef_[7] + \
                                model.intercept_
print(pred)



# 회귀분석을 위한 머신러닝 모델의 평가함수

# - score 메소드를 활용 : R2(결정계수)
#                      : 데이터에 관계없이 동일한 범위 사용

# - 평균절대오차 : 실제 정답과 모델이 예측한 값의 차이를 절대값으로 평균
#                (머신러닝 모델이 예측한 값의 신뢰 범위 내에서...)
# - 평균절대오차비율 : 실제 정답과 모델이 예측한 값의 비율 차이를 절대값으로 평균

# - 평균제곱오차 : 실제 정답과 모델이 예측한 값의 차이의 제곱값 평균
#                 (머신러닝/딥러닝 모델의 오차 값을 계산할 때 사용)

# R2 (결정계수)
from sklearn.metrics import r2_score
# 평균절대오차
from sklearn.metrics import mean_absolute_error
# 평균절대오차비율
from sklearn.metrics import mean_absolute_percentage_error
# 
from sklearn.metrics import mean_squared_error

# (공통적으로,) 평가를 위해서는 머신러닝 모델이 예측한 값이 필요함.
pred = model.predict(X_train)
print(y_train.describe())

# 평균절대오차
mae = mean_absolute_error(y_train, pred)
print(f'MAE : {mae}')

# 평균절대오차비율
mape = mean_absolute_percentage_error(y_train, pred)
print(f'MAPE : {mape}')
















