# -*- coding: utf-8 -*-

# 스태킹 모델
    # 앙상블 리뷰
    # : 다수개의 모델 예측값 -> 취합해 "다수결/평균"으로 예측하는 모델
    # : 일반화 높이기 위해 사용 (예측 성능 분산 감소 목적)
# : 다수개의 모델 예측값 -> 바탕으로 학습(새로운 모델)


import pandas as pd
pd.options.display.max_columns = 100
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target)


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(penalty='l2',
                        class_weight='balanced',
                        random_state=1,
                        solver='lbfgs',
                        max_iter=100,
                        n_jobs=-1).fit(X_train_scaled, y_train)
kn = KNeighborsClassifier(n_neighbors=5,
                          n_jobs=-1).fit(X_train_scaled, y_train)
dt = DecisionTreeClassifier(max_depth=3,
                            random_state=1).fit(X_train_scaled, y_train)

lr_train_score = lr.score(X_train_scaled, y_train)
print(f'lr TRAIN score : {lr_train_score}')
lr_test_score = lr.score(X_test_scaled, y_test)
print(f'lr TEST score : {lr_test_score}\n')
kn_train_score = kn.score(X_train_scaled, y_train)
print(f'kn TRAIN score : {kn_train_score}')
kn_test_score = kn.score(X_test_scaled, y_test)
print(f'kn TEST score : {kn_test_score}\n')
dt_train_score = dt.score(X_train_scaled, y_train)
print(f'dt TRAIN score : {dt_train_score}')
dt_test_score = lr.score(X_test_scaled, y_test)
print(f'dt TEST score : {dt_test_score}\n')


# 스태킹 모델 구현
# 1. 앙상블 구성 모델의 각 예측 결과 취합
import numpy as np

pred_lr = lr.predict(X_train_scaled)
pred_kn = kn.predict(X_train_scaled)
pred_dt = dt.predict(X_train_scaled)

pred_stack = np.array([pred_lr, pred_kn, pred_dt])
# print(pred_stack)
# print(pred_stack.shape)

pred_stack = pred_stack.T
# print(pred_stack)
# print(pred_stack.shape)

final_model = LogisticRegression(random_state=1).fit(pred_stack, y_train)
score = final_model.score(pred_stack, y_train)
print(f'학습 final model : {score}')

pred_lr = lr.predict(X_test_scaled)
pred_kn = kn.predict(X_test_scaled)
pred_dt = dt.predict(X_test_scaled)
pred_stack = np.array([pred_lr, pred_kn, pred_dt]).T
final_model = LogisticRegression(random_state=1).fit(pred_stack, y_test)
score = final_model.score(pred_stack, y_test)
print(f'테스트 final model : {score}')










