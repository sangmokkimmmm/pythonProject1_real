print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names  # data set 불러오기

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)         # 주어진 data set을 어떻게 test와 train data로 나눌 것인가의 작업
# train_test_polit(arrays, test_size, train_size, random_state, shuffle, stratify
# array = x,y
# test_size = 테스트 데이터셋의 비율이나 갯수 (default=0.25)
# train_size 학습 데이터셋의 비율이나 갯수 (default 1-test_size)
# random_state = data 분할시 셔플이 이루어지는데 이를 위한 시드값 ,
# shuffle = 셔플여부설정 (default = true)
# stratify : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.
# [출처] [Python] sklearn의 train_test_split() 사용법|작성자 Paris Lee




# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)    # classifier() : 분류하기 위한 classifier 생성 + .fit() : test 하기
# svm = support vector machine, svm.svc는 classifier 의 한 종류
# https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-3%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0-SVM-%EC%8B%A4%EC%8A%B5 참고
# kernel paprameter를 linear로 한다 <= 단순하게 분류할 수 있기 때문
# C : 작게하면 training data의 분류를 부정확하게 하는 대신, margin을 크게한다. C를 크게하면 margin을 작게하는 대신 training data의 분류를 정확하게 한다.
# 즉, Noise가 많은 data = 경계가 불분명한 data는 C를 작게하는 것이 좋고, noise가 별로 없는 데이터는 C를 크게 해야한다.
# if, linear한 선으로 claasify가 불가능한 경우. kernel trick을 활요해야한다. ex) 원으로 boundary를 결정해야하는 경우, rbf 함수를 사용.
# C와 gamma를 다양하게 조정해서 최적의 decison boundary를 찾아야한다. 하나하나 해봐야한다. 하지만 sklearn에서는 gridsearchCV라는 method가 이걸 해준다.
######################################
# gridsearchCV 사용법 예시\
# param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
#
# # Make grid search classifier
# clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
#
# # Train the classifier
# clf_grid.fit(X_train, y_train)
# clf = grid.best_estimator_()
# print("Best Parameters:\n", clf_grid.best_params_)
# print("Best Estimators:\n", clf_grid.best_estimator_)

# print("Displaying decision function for best estimator.")
# Plot decision function on training and test data
# plot_decision_function(X_train, y_train, X_test, y_test, clf_grid)



np.set_printoptions(precision=2) # numpy float 출력 옵션 변경 .이하 수 가 2개.



# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
# disp = display 즉, 보여주는 함수이다.
# https://ichi.pro/ko/sklearnui-confusion-matrix-mich-plot-confusion-matrix-hamsuleul-ilg-go-haeseoghaneun-bangbeob-43639744954561 참고


plt.show()