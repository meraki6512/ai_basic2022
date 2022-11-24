# -*- coding: utf-8 -*-

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)

print(X[:10])
print(y[:10])

import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1])
plt.show()


# 군집분석 (3) DBSCAN 알고리즘
# - 데이터간 밀도를 이용해 군집 형성.
# - 군집의 크기가 자동으로 결정됨. (n_clusters설정필요없음)

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_cluster = db.fit_predict(X)



plt.scatter(X[y_cluster==0, 0], 
            X[y_cluster==0, 1],
            s=50, marker='s', label='Cluster1',
            c='lightgreen')

plt.scatter(X[y_cluster==1, 0], 
            X[y_cluster==1, 1],
            s=50, marker='o', label='Cluster2',
            c='orange')

plt.legend()
plt.grid()
plt.show()








