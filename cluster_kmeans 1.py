# -*- coding: utf-8 -*-

# 비지도학습

# - 정답데이터가 제공되지 않는 데이터에 대한 학습을 처리하는 기법
# - 결과는 주관적인 판단으로 처리함. 
# - 결과를 100% 신뢰할 수 없음!
# - 매 실행마다 결과가 바뀔 수 있음!

# 1) 차원축소
    # - 특정 컬럼 추출
    # - 시각화를 위해 사용되는 경우가 많음.
    # - 예) 소셜네트워크 분석, 이미지 분석, RGB 컬러
# 2) 군집분석
    # - 데이터(샘플)의 유사성 비교해 동일한 특성으로 구성된 샘플들을 하나의 군집으로 구성.
    
    
# 군집분석    
# - 데이터의 "클러스터링" 과정    
# - 지도학습에서 군집분석의 결과를 활용하는 방법

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


# KMeans
# - 최근접 이웃 알고리즘과 유사 
# - 가장 많이 사용되는 군집분석 클래스 : 단순, 변경용이

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            random_state=0)
km.fit(X) #n_cluster만큼 포인터 지정해 최적의 위치 찾는 과정

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





















