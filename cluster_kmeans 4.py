# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.datasets import load_breast_cancer

pd.options.display.max_columns = 100

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target, name = 'target')

print(X.head())
print(X.info())
print(X.describe())
print(y.head())





X_part = X[['radius error', 'compactness error', 'concavity error']]
from sklearn.cluster import KMeans

values = []
for i in range(1, 15):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X_part)    
    values.append(km.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,15), values, marker='o')
plt.xlabel('number of cluster')
plt.ylabel('inertia_')
plt.show()

# 5개의 군집이 적당한 것 같다..!

from sklearn.cluster import KMeans
km = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
km.fit(X_part)
X['cluster_result'] = km.predict(X_part)

del X['radius error']
del X['compactness error']
del X['concavity error']
print(X.info())





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0,
                           random_state=0,
                           class_weight='balanced').fit(X_train_scaled, y_train)

train_score = model.score(X_train_scaled, y_train)
print(f'train score = {train_score}')
test_score = model.score(X_test_scaled, y_test)
print(f'test score = {test_score}')

print(f'coef_ : \n{model.coef_}') # 가중치가 높을수록 중요도가 높음.
#10, 15, 16 #번째 컬럼 가중치가 상대적으로 작음
print(X.info())
#['radius error', 'compactness error', 'concavity error']


# 성능이 좋아진 건 아니지만 이러한 방법을 소개하기위해 설명.













