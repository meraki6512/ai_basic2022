# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

print(X[:10])
print(y[:10])


import matplotlib.pyplot as plt
# plt.scatter(X[:,0], X[:,1])
# plt.show()


from sklearn.cluster import KMeans

# 최적의 군집(클러스터) 개수
# - 엘로우 기법
values = []
for i in range(1,11):
    
    km = KMeans(n_clusters=i,
            #init='random',
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0).fit(X) #n_cluster만큼 포인터 지정해 최적의 위치 찾는 과정
    
    # inertia_ : 각 클래스의 SSE 반환
    values.append(km.inertia_)
    
print(values)

plt.plot(range(1,11), values, marker='o')
plt.xlabel('numbers of cluster')
plt.ylabel('inertia_')
plt.show()

y_cluster = km.predict(X)
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

plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1], 
            s=100, marker='*', label='Center',
            c='red')

plt.legend()
plt.grid()
plt.show()

###########################################################

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

###########################################################

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)

print(X[:10])
print(y[:10])


from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_cluster = db.fit_predict(X)

# 시각화 
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
