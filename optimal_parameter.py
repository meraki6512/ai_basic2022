# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    stratify=y, 
                                                    random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                      test_size = 0.1,
                                                      stratify=y_train,
                                                      random_state=1)
print(f'TRAIN : {len(X_train)}, VALID : {len(X_valid)}, TEST : {len(X_test)}')




# 최적의 하이퍼파라미터 찾기
# (성능 올리는 베스트 방법 : 전처리) 이미 했다고 생각하고
# 하이퍼 파라미터 조절 !!

# ensemble review
# - RandomForest 최대한 강하게 배울 수 있도록, 제약 DOWN 
# - GradientBoosting 학습을 되도록 못하도록, 제약 UP

best_score = None
best_param = {}

# GradientBoostingClassifier()의 하이퍼파라미터
# learning_rate, n_estimators, subsample, max_depth, random_state ... 

for lr in [0.1, 0.2, 0.3, 1., 0.01]:
    for md in [1,2,3]:
        for ne in [100, 200, 300, 10, 50]:
            
            model = GradientBoostingClassifier(
                learning_rate=lr,
                max_depth=md,
                n_estimators=ne,
                random_state=1).fit(X_train, y_train)
            
            # score = model.score(X_test, y_test) # test 데이터로 평가 시... 학습은 잘했어도 테스트에서 안 좋은 점수가 나와 버려질 수 있음.
            score = model.score(X_valid, y_valid) # valid 데이터 (검증 데이터) 분할 방법에 따라 점수가 달라질 수 있음
            if not best_score or (best_score and score > best_score):
                best_score = score
                best_param = {'learning_rate' : lr,
                              'max_depth' : md,
                              'n_estimators' : ne,
                              'random_state' : 1}

print(f'best_score : {best_score}')
print(f'best_param : \n{best_param}')



# 모델 생성, 학습, 평가
best_model = GradientBoostingClassifier(
    learning_rate = best_param['learning_rate'],
    max_depth = best_param['max_depth'],
    n_estimators= best_param['n_estimators'],
    random_state = best_param['random_state']).fit(X_train, y_train)

print(f'Score(TRAIN) : {best_model.score(X_train, y_train)}')
print(f'Score(TEST) : {best_model.score(X_test, y_test)}')





# 최적의 하이퍼 파라미터 찾을 때 ... 

# 1. test 데이터로 찾는 경우 
    # : 학습에서 좋은 성능인데 테스트에서 안좋은 성능을 보여 해당 파라미터를 날릴 수 있음.
# 2. valid 데이터로 찾는 경우
    # : 검증 데이터의 분할에 따라 성능 차이가 있을 수 있음.
# 3. -> 교차검증!!!










