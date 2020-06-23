import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from problem_3.load_data import load_data_SVC


# SVM Classifier
def problem_3_SVM():
    try:
        os.remove('./results/problem_3_SVM.txt')
    except OSError:
        pass

    # Load data
    X_train, Y_train, X_test, Y_test = load_data_SVC('MNIST')

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    for kernel in kernels:
        clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, gamma='scale'))
        clf.fit(X_train, Y_train)
        accuracy = clf.score(X_test, Y_test)

        print(
            f"The SVM classification accuracy with {kernel} Kernel is: {accuracy * 100:0.2f}%")
        with open('./results/problem_3_SVM.txt', "a") as file:
            file.write(
                f"The SVM classification accuracy with {kernel} Kernel is: {accuracy * 100:0.2f}%\n")
