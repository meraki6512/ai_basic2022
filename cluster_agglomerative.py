# -*- coding: utf-8 -*-

# 군집분석 - (2) 병합군집
# 다수개의 소규모 군집 랜덤적으로 생성 후 # 병합

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

print(X[:10])
print(y[:10])

# 시각화
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], 
            c = 'white',
            marker = 'o',
            edgecolors='black',
            s = 50)
plt.grid()
plt.show()

# 병합군집 클래스 : AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3)
y_cluster = ac.fit_predict(X) # 소규모 군집을 계속해서 병합하다가 n_cluster에 도달
print(y_cluster)


# 시각화
plt.scatter(X[y_cluster==0, 0], 
            X[y_cluster==0, 1],
            s=50, marker='s', label='Cluster1',
            c='lightgreen')

plt.scatter(X[y_cluster==1, 0], 
            X[y_cluster==1, 1],
            s=50, marker='o', label='Cluster2',
            c='orange')

plt.scatter(X[y_cluster==2, 0], 
            X[y_cluster==2, 1],
            s=50, marker='v', label='Cluster3',
            c='lightblue')

plt.legend()
plt.grid()
plt.show()




# 군집 분석을 할 때도 "스케일링"처리는 필수다!


















