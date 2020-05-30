import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from problem_3.load_data import load_data_SVC


# SVM Classifier
def problem_3_SVM():
    os.remove('./results/problem_3_SVM.txt')

    # Load data
    X_train, Y_train, X_test, Y_test = load_data('MNIST')

    # Gaussian Kernel
    gauss_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale'))
    gauss_clf.fit(X_train, Y_train)
    gauss_accuracy = gauss_clf.score(X_test, Y_test)

    print(
        f"The SVM classification accuracy with Gaussian Kernel with {noise}% label noise is: {gauss_accuracy * 100:0.2f}%")
    with open('./results/problem_3_SVM.txt', "a") as file:
        file.write(
            f"The SVM classification accuracy with Gaussian Kernel with {noise}% label noise is: {gauss_accuracy * 100:0.2f}%\n")

    # Polynomial Kernel
    poly_clf = make_pipeline(StandardScaler(), SVC(kernel='poly', gamma='scale'))
    poly_clf.fit(X_train, Y_train)
    poly_accuracy = poly_clf.score(X_test, Y_test)

    print(
        f"The SVM classification accuracy with Polynomial Kernel with {noise}% label noise is: {poly_accuracy * 100:0.2f}%")
    with open('./results/problem_3_SVM.txt', "a") as file:
        file.write(
            f"The SVM classification accuracy with Polynomial Kernel with {noise}% label noise is: {poly_accuracy * 100:0.2f}%\n")
