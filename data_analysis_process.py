# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:19:56 2022

@author: 82103
"""

# 데이터 분석 (머신러닝 / 딥러닝) 수행하는 과정



#1. 데이터의 적재 (로딩)
import pandas as pd
# - 유방암 데이터 셋 (분류용 데이터 셋, 암의 악성 여부)
from sklearn.datasets import load_breast_cancer
# 사이킷 런에서 제공하는 오픈 데이터
data =  load_breast_cancer()

print(data.keys())

# 설명변수 (X) : 특정 종속변수 y를 유추하기 위한 데이터 셋
X = pd.DataFrame(data.data, columns = data.feature_names)
# 종속변수 (y) : 정답 데이터, Label
y = pd.Series(data.target)





#2. 데이터의 관찰(탐색) EDA

# 설명변수 (X)에 대한 데이터 탐색

# - 데이터의 개수,  컬럼의 개수
# - 각 컬럼 데이터에 대한 결측 데이터의 존재 유무
# - 각 컬럼 데이터의 데이터 타입 (반드시 수치형의 데이터만 머신러닝에 활용할 수 있음.)
print(X.info())

# pandas 라이브러리 옵션 설정 
# - 컬럼 수 제어 (기본 출력 컬럼 수 4~5개)
pd.options.display.max_columns = 30

# 데이터를 구성하는 각 컬럼에 대해 기초 통계 정보 확인
# - count
# - mean, std
# - min, max
# - 4분위값
# 이때, 각 컬럼의 스케일 체크가 포인트다.
print(X.describe())

# 종속변수 y
print(y)

# 종속변수의 값이 범주형인 경우
# 범주형 값의 확인 및 개수 체크
print(y.value_counts())

# 값의 개수 비율이 중요하다.
# 그 이유!! 
# - 머신러닝으로 예측하려는 데이터는 데이터의 비중 작음.
# - 극단적인 케이스로 ...
#   악성 1%, 악성X 99%일 때,
#   '악성이 아님'으로 예측만 수행해도 99%의 정확도를 갖게 됨.
# 데이터의 비율에서 많은 차이가 발생하는 경우
#  : 오버샘플링 (큰 비율 데이터에 개수를 맞춤.) / 언더샘플링 (작은 비율 데이터에 맞춰 데이터 버림.)
print(y.value_counts()/len(y))






# 데이터 전처리 - 데이터 분할 이후에 수행함. 
# ? 예를 들어, 이 부분에서 Scaling 처리를 미리 한다면,
# 학습의 성능은 올라가지만 실전에서 좋지 않은 결과의 모델이 생성될 가능성이 높다.

# - 데이터 전처리 : "학습" 데이터에 대해서 수행
# - "테스트" 데이터 : 학습 데이터에 반영된 결과를 수행

# 3. 데이터의 분할
# - 학습 데이터(7~80%)와 테스트 데이터(2~30%)로 분할 (머신러닝의 케이스)
# - 학습 데이터(70%)와 검증 데이터(10%)와 테스트 데이터(20%)로 분할 (딥러닝의 케이스)
#   (딥러닝의 경우, 부분 배치 학습을 수행해, 점진적으로 학습량을 늘려나가는 경우가 많음.)
#   (검증 데이터 -> 중간 점검의 의미로, 부분 배치 학습에 활용. )

from sklearn.model_selection import train_test_split

# 사용 예제
# X_train, X_test, y_train, y_test 
#   = train_test_split(X, y, 
#                            test_size = 테스트데이터비율(0.2), # 또는
#                            train_size = 테스트데이터비율(0.8), 
#                            stratify = y, # 범주형 데이터인 경우만
#                            random_state = 임의의 정수값)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(len(X_train), len(X_test))

print(y_train[:5])
# train_test_split 함수의 random_state 매개변수는 
# 데이터의 분할 값이 항상 동일하도록 유지하는 역할을 함.
# - 머신러닝 알고리즘에 사용되는 하이퍼 파라미터를 테스트할 때,
#   데이터는 고정하고 머신러닝의 학습 방법만을 제어해가면서 성능 향상의 정도를 테스트할 수 있음.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=81)
print(y_train[:5])

# stratify: 데이터가 분류형 데이터 셋일 경우에만 사용
#           (즉, y 데이터가 범주형인 경우)
#           random_state에 관계없이
#           각 범주형 값의 비율을 유지하면서 데이터를 분할하는 역할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)
# y 전체 데이터의 비율
# 1    0.627417
# 0    0.372583
# dtype: float64
print(y.value_counts()/len(y))
print(y_train.value_counts()/len(y_train))





# 4. 데이터 전처리
# - 데이터 스케일 처리 (MinMax, Standard, Robust(이상치))
# - 인코딩 처리 (라벨 인코딩, 원핫 인코딩)
# - 차원 축소 
# - 특성 공학 ...
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# 전처리 과정의 데이터 학습은 학습 데이터를 기준으로 수행.
# - 학습 데이터의 최소 / 최대값을 기준으로 스케일링을 수행 준비 
scaler.fit(X_train)

# - 학습 데이터의 스케일링 처리 수행
X_train = scaler.transform(X_train)

# - 테스트 데이터는 학습 데이터를 기준으로 변환 과정만 수행
#    (실제 데이터와 같은 다른 범위의 값들을 대비하기 위해)
X_test = scaler.transform(X_test)







# 5. 머신러닝 모델 구축
from sklearn.neighbors import KNeighborsClassifier

# - 머신러닝 객체 생성
# - 각 머신러닝 알고리즘에 해당하는 하이퍼 파라미터의 제어가 필수적임.
#   - n_neighbors: 최근접 이웃 알고리즘에 사용될 이웃의 수; default=5; (중요!!)
#   - n_jobs: 병렬적인 연산을 해도 무방할 시 사용될 코어의 수; default=None; -1값을 주면 가능한 모든 프로세서 사용을 뜻함. 
model = KNeighborsClassifier(n_neighbors=11, n_jobs=-1);

# 머신러닝 모델 객체 학습
# - fit 메소드 사용
#   모델 객체. fit(X, y)
# - 사이킷런의 모든 머신러닝 클래스는 fit 메소드의 매개변수로
#   X, y를 입력받음.
#   (X는 반드시 2차원의 데이터 셋임. - pandas의 DataFrame, python의 list ...)
#   (y는 반드시 1차원의 데이터 셋임. - pandas의 Series, python의 list ...)
model.fit(X_train, y_train)

# 학습이 완료된 머신러닝 모델의 객체 평가
# - score 메소드 사용
#   모델 객체.score(X, y)
#   : 입력된 X를 사용하여 예측을 수행하고
#     예측된 값을 입력된 y와 비교하여 평가 결과를 반환함.
# 
# - 주의사항 !!!
# 머신러닝 클래스의 타입이 ...
# 분류형이라면, score 메소드는 정확도(전체데이터 중 정답인 데이터 비율) 반환
# 회귀형이라면, score 메소드는 결정 계수(x<=1의 값을 가짐; 1일 때, 100% 예측 일치) 반환
score = model.score(X_train, y_train)
print(f'Train: {score}')

score = model.score(X_test, y_test)
print(f'Train: {score}')

# 학습된 머신러닝 모델을 사용하여 예측 수행
# - predict 메소드 사용
# model.predict(X)
# - 주의: 예측할 데이터 X(설명변수)는 반드시 2차원으로 입력되어야 한다.
pred = model.predict(X_train[:2])
print(pred)
print(y_train[:2])

pred = model.predict(X_test[-2:])
print(pred)
print(y_test[-2:])

# 학습된 머신러닝 모델이 분류형인 경우, 
# 확률 값으로 예측 가능 (단, 일부 클래스에서는 제공X)
# - predict_proba 메소드 사용
# model.predict_proba(X)
# - 주의: 마찬가지로, X는 2차원 입력
# 예측 확률의 기준선의 개념을 도입: 일정 확률 이상으로 예측하는 경우에만 그 결과를 수용하는 방법 사용
proba = model.predict_proba(X_test[-2:])
print(proba)

#최근접 이웃 알고리즘
#n_neighbors의 default값이 5이므로 [0.4 0.6], [0.8 0.2], [0. 1.] 등의 값이 나오고 있는 것
proba = model.predict_proba(X_test)
print(proba)




















