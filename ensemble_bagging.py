# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:14:05 2022

@author: 82103
"""

#*** (앙상블) 중요한 ***

# Ensemble (예측 수행 방법(론))
#
# 1) 취합 
#   : 각 모델이 독립적 (연관성x) (분류분석-다수결, 회귀분석-평균)
#   -> 적절한 수준의 과적합을 수행할 필요가 있음.
#   : (학습 및 예측 수행) 속도가 빠름. (병렬처리 가능)
#   - Voting, Bagging, RandomForest
#
# 2) 부스팅
#   : 각 모델이 서로 선형적으로 연결 (앞 모델이 다음 모델에 영향)
#   -> 강한 제약을 설정해(제어를 적절히 해), 점진적인 성능을 도모해야함.
#   : (학습 및 예측 수행) 속도가 느림. (병렬처리 불가능)
#   - AdaBoosting, GradientBoosting, XGBoost, LightGBM






# Bagging

import pandas as pd
pd.options.display.max_columns = 100

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target)

y.head()
y.tail()
y.value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)


#앙상블 클래스 로딩
from sklearn.ensemble import BaggingClassifier

#앙상블을 구현하기 위한 내부 모델의 클래스 로딩 
from sklearn.tree import DecisionTreeClassifier

base_estimator = DecisionTreeClassifier(random_state=1)

model = BaggingClassifier(base_estimator=base_estimator,
                          n_estimators=50,
                          max_samples = 0.3,
                          max_features = 0.3,
                          n_jobs = -1,
                          random_state = 1)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(f'Score (Train): {score}')
score = model.score(X_test, y_test)
print(f'Score (Test): {score}')

#앙상블 모델의 예측 결과
pred = model.predict(X_test[50:51])
print(f'Predict: {pred}')













































