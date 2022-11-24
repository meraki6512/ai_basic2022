# -*- coding: utf-8 -*-

import pandas as pd
pd.options.display.max_rows=100
pd.options.display.max_columns=100

# 1. 데이터 적재
fname_input = './data/spaceship_titanic.csv'
data = pd.read_csv(fname_input)

# 2. EDA 수행
# - ID가 있는 것 같음 (PassengerId)
# - 종속변수의 타입이 문자열? 진리형?
print(data.head())

# - 데이터의 전체 개수 : 8693
# - 데이터 타입이 다양함 : float, object, bool
# - 결측 데이터가 존재함
print(data.info())

# 결측 데이터의 개수 확인
print(data.isnull().sum())

# 결측 데이터 제거
# - 레코드 단위로 제거
data = data.dropna()
print(data.info())

# 고유 값 저장 컬럼 확인
print(data.PassengerId.value_counts())
print(data.Name.value_counts())

columns = data.columns
columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
       'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 
       'VRDeck', 'Transported']
data = data[columns]
print(data.info())

# 종속변수의 값 변경 (Transported 컬럼)
print(data.Transported)

# apply 메소드 사용
# data.Transported = 1 if data.Transported else 0


# 데이터 전처리 1
# - 원핫 인코딩
obj_columns = [cname for cname in data.columns if data[cname].dtype == 'object']
num_columns = [cname for cname in data.columns if data[cname].dtype in ['int64','float64','bool']]

# - 원핫 인코더 import
from sklearn.preprocessing import OneHotEncoder

# - 원핫 인코더는 다수개의 컬럼에 대해서 동시에 전처리가 가능함
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(data[obj_columns])

# - 원핫 인코딩을 적용한 결과 컬럼의 개수가 증가한 것을 확인할 수 있음
encoder_output = encoder.transform(data[obj_columns])
print(encoder_output.shape)

# 인코딩의 결과를 데이터프레임으로 생성
df_encoded = pd.DataFrame(encoder_output)
print(df_encoded.info())

# 인덱스 정보의 불일치로 데이터 결합에서 문제가 발생함
print(data.info())

data.reset_index(inplace=True)
print(data.info())

# 두 개의 데이터프레임을 하나로 결합
data = pd.concat([data[num_columns], df_encoded], axis=1)
print(data.info())


# 기본 전처리 과정이 끝났으므로 
# 설명변수 / 종속변수로 분리
X = data.drop(columns=['Transported'])
y = data.Transported

print(X.columns)
print(y.head())

# 데이터 분할
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,
                                                 test_size=0.3,
                                                 stratify=y,
                                                 random_state=11)

print(X_train.shape, X_test.shape)


# 전처리 2
# - 수치 데이터에 대한 스케일링 처리
# - MinMaxScaler 클래스는 컬럼내의 최대 값을 1로 
#  최소값을 0으로 전처리하는 클래스
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# - 종속변수가 포함되어 마지막 종속변수의 컬럼명 제거
num_columns = num_columns[:-1]

# - 스케일링을 위한 전처리 클래스 (MinMax, Standard, Robust) 들은
#   다수개의 컬럼에 대해서 1개의 스케일러가 처리할 수 있음
scaler.fit(X_train[num_columns])

temp = scaler.transform(X_train[num_columns])

print(X_train[num_columns].head(5))
print(temp[:5])


# 학습 데이터에 스케일 처리된 결과를 대입
X_train[num_columns] = temp
print(X_train[num_columns].head(10))

# 테스트 데이터에 스케일 처리된 결과를 대입
print(X_test[num_columns].head())
X_test[num_columns] = scaler.transform(X_test[num_columns])
print(X_test[num_columns].head())


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

model_knn = KNeighborsClassifier(n_neighbors=5,
                                n_jobs=-1).fit(X_train,y_train)

model_rf = RandomForestClassifier(n_estimators=100,
                                 max_depth=None,
                                 class_weight={False:0.3,True:0.7},
                                 n_jobs=-1,
                                 random_state=11).fit(X_train,y_train)

model_gb = GradientBoostingClassifier(n_estimators=200,
                                     max_depth=1,
                                     subsample=0.5,
                                     random_state=11).fit(X_train,y_train)

# - 평가를 위한 함수 import
from sklearn.metrics import precision_score

# - 평가 값을 각 모델 별로 테스트 데이터를 기준으로 추출
score_knn = precision_score(y_test, model_knn.predict(X_test))
score_rf = precision_score(y_test, model_rf.predict(X_test))
score_gb = precision_score(y_test, model_gb.predict(X_test))

print(f'score_knn : {score_knn}')
print(f'score_rf : {score_rf}')
print(f'score_gb : {score_gb}')

# score_knn : 0.7504761904761905
# score_rf : 0.7947725072604066
# score_gb : 0.775647171620326

# 모델 선택 : 랜덤포레스트 모델을 사용함
# 선정 이유 : 하이퍼 파라메터를 추가적으로 제어하여 결과를 개선할 수 있지만
#            현재 모델의 평가 결과를 기반으로 가장 오차가 적은 랜덤포레스트
#            모델을 선정함
best_model = model_rf

# - 정확도
score_acc = best_model.score(X_test, y_test)
# - 정밀도
score_ps = precision_score(y_test, best_model.predict(X_test))

print(f'score_acc : {score_acc}')
print(f'score_ps : {score_ps}')

# score_acc : 0.7840565085771948
# score_ps : 0.786144578313253

from sklearn.metrics import classification_report
cr = classification_report(y_test, best_model.predict(X_test))
print(cr)

#               precision    recall  f1-score   support

#        False       0.80      0.76      0.78       984
#         True       0.77      0.81      0.79       998

#     accuracy                           0.78      1982
#    macro avg       0.78      0.78      0.78      1982
# weighted avg       0.78      0.78      0.78      1982

# 모델 분석
# - 현재 모델링한 머신러닝 모델의 결과 분석은 정밀도의 기준값에 따라 변경될 수 있지만
#   일반적인 판단 기준으로 성능이 낮음 (적어도 0.85 이상은 되어야 신뢰성이 있음)
# - 다만 classification_report 함수의 결과를 확인하면 종속변수 2가지 케이스에 대해서
#   고른 성적을 보이는 것을 확인할 수 있음
#   (목표로 하는 종속변수의 값에 대한 가중치 조정을 시도하면 특정 값의 정밀도는
#   많은 향상을 보일 것으로 예상됨)


























