# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 07:37:04 2022

@author: 82103
"""

# 앙상블 (Ensemble)
# 다수개의 모델을 결합하여 예측을 할 수 있는 결합 모델
# 일반적으로, 다수결/평균의 원칙 적용
# 베스트 평가 성적을 반환하지는 않음. (일반화에 목적을 둔다.)

# 앙상블을 구현하는 방법
# 1. 취합
#   - 앙상블 구성 모델들(estimators)이 각각 독립적으로 동작
#   - Voting, Bagging, RandomForest
#   - 각각의 구성 모델은 일정 수준의 예측 성능을 달성해야한다.
# 2. 부스팅
#   - 앙상블 구성 모델들이 선형적으로 결합.. 점진적으로 성능 향상
#   - AdaBoosting, GradientBoosting
#   - XGBoost, LightGBM
#   - 각각의 구성 모델은 강한 제약을 설정해 학습을 제어한다.
#   - AdaBoosing, GradientBoosting: 학습 및 예측의 속도가 느림. (병렬처리 불가능)

import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)




# (3) RandomForest
# Bagging + DecisionTree
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,
                               max_depth = None,
                               max_samples = 0.7,
                               max_features = 0.7,
                               random_state=1,
                               n_jobs=-1)

model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)



























