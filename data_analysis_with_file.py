# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:19:15 2022

@author: 82103
"""

import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 10
fname_input = './data/titanic.csv'

data = pd.read_csv(fname_input, 
                   header='infer', 
                   sep=',')

print(data.head())
# dtypes: float64(2), int64(5), object(5)
print(data.info())
print(data.describe())

# 문자열 데이터에 대해서 기초 통계정보 확인
# - count: 결측이 아닌 데이터의 개수
# - unique: 중복을 제외한 데이터의 개수
print(data.describe(include=object))

# 수치형 데이터에도 ID 성격의 데이터가 존재할 수 있음.
# - 데이터 개수가 모두 1.
# - 하단의 Length 정보와 전체 데이터 개수가 일치함.
print(data.value_counts())
print(data.PassengerId.value_counts())

# 불필요한 정보는를 확인 후 제거할 필요가 있다.
# - ID 성격의 데이터, 즉, 각 데이터의 빈도가 적으면 주의할 것. (ID 성격)
# - 데이터 수가 적어도 주의할 것.
#   ['PassengerId', 'Name', 'Ticket', 'Cabin']
data2 = data.drop(
    columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

# 결측 데이터는 항상 체크한다.
print(data2.info())
print(data2.isnull().sum())
# - 처리
# 방법1. 해당 열을 제거한다.
# 방법2. 해당 행을 제거한다.
# 방법3. 기초 통계로 결측치를 대체한다.  
    #(평균, 중심값, 최빈값 등/ 최빈값)
    #신뢰성?
# 방법4. 지도학습 기반의 머신러닝 모델을 구축하여 예측한 값으로 결측 데이터를 대체한다.
    # - Age 컬럼: 결측치가 아닌 데이터가 월등히 더 많으므로 학습 고려 대상이다.
# 방법5. 준지도학습, 비지도학습 기반의 머신러닝 예측한 값으로 결측 데이터를 대체한다.
    # ex) 클러스터링
    # - Cabin 컬럼: 결측치 데이터가 월등히 더 많음.
    

data3 = data2.dropna(subset=['Age','Embarked'])
print(data3.info())
print(data3.isnull().sum())


# 문자열 데이터 Sex, Embarked
print(data.Sex.value_counts())
print(data.Embarked.value_counts())

# Q. 각 데이터가 학습, 테스터 데이터에 각각 잘 들어갈까?
#   - 수치 데이터와 연계하는 방법
#   - 데이터 분할 전에 문자열 전처리 하는 것이 원칙

# 문자열 전처리 방식
# 1. 라벨 인코딩 : 정수와 단순 매칭
    # y(정답데이터)가 문자열로 되어있는 경우에 사용 가능.
    # X(설명변수)에는 잘 사용하지 않음.
        #X: 2차원배열: 값의 유실 또는 배가 적용되기 때문.
# 2. 원핫 인코딩 : unique한 문자열의 개수만큼 컬럼 생성해, 하나의 컬럼에만 가중치 적용.
    # X 전처리에 주로 사용.
    # 메모리 낭비가 극심하다는 단점이 있다. (unique 값의 case를 최소화할 필요가 있음.)

X = data3.iloc[:, 1:]
y = data3.Survived

print(X['Pclass'].dtype)
print(X['Age'].dtype)
print(X['Sex'].dtype)

# 수치형 데이터와 문자열 데이터를 분리하는 것이 편리함.
    # (어차피, 두 데이터의 전처리 과정은 독립적.)
X_num = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
X_num = X[X_num]

X_obj = [cname for cname in X.columns if X[cname].dtype not in ['int64', 'float64']]
X_obj = X[X_obj]

print(X_num.info())
print(X_obj.info())


from sklearn.preprocessing import OneHotEncoder
# onehotencoder의 주요 parameter
    # - sparse: 희소행렬 생성 여부
    # - handle_unknown: {'ignore', 'error'}
encoder = OneHotEncoder(sparse=False, 
                        handle_unknown='ignore')


#사이킷런의 전처리 클래스는 학습용 클래스와 유사하다.
    # 1. fit 메소드: 전처리 과정에 필요한 정보 수집
    # 2. transform 메소드: 값 할당
        # 1+2: fit_transform 메소드
X_obj_encoded = encoder.fit_transform(X_obj)

print(X_obj.head())
print(X_obj_encoded[:5])

# 주의사항
#   - 사이킷런의 모든 전처리 클래스는 
#     transform의 결과가 numpy 배열로 반환
#    (pandas의 데이터프레임이 아님!!!)

print(encoder.categories_)
print(encoder.feature_names_in_)

X_obj_encoded = pd.DataFrame(X_obj_encoded, columns = ['s_f', 's_m', 'e_C', 'e_Q', 'e_S'])
print(X_obj_encoded)



#전처리된 데이터를 병합해 X(설명변수)를 생성한다.
print(X_num.info())
print(X_obj_encoded.info())
# 결측치를 제거했기때문에 인덱스를 다시 초기화해줘야한다.
X_num.reset_index(inplace=True)
X_obj.reset_index(inplace=True)
# - concat 메소드
X = pd.concat([X_num, X_obj_encoded], axis=1)
print(X.info())
print(y.value_counts())
print(y.value_counts()/len(y))






# 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    stratify = y,
                                                    random_state=0)


# 데이터 학습
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

model = RandomForestClassifier(n_estimators=100,
                               max_depth=None,
                               max_samples=1.0,
                               class_weight='balanced',
                               n_jobs=-1,
                               random_state=0)

model.fit(X_train, y_train)
score = model.score(X_train, y_train)
print(f'Score(Test): {score}')
score = model.score(X_test, y_test)
print(f'Score(Test): {score}')
















