import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from problem_1.load_data import load_data


# SVM Classifier
def problem_1_SVM():
    try:
        os.remove('./results/problem_1_SVM.txt')
    except OSError:
        pass

    noise_level = [0, 20, 40]
    for noise in noise_level:
        # Load data
        X_train, Y_train, X_test, Y_test = load_data(f'/2class-Synthetic/data_noise_{noise}.txt')

        kernels = ['linear', 'poly', 'rbf', 'sigmoid']

        for kernel in kernels:
            clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, gamma='scale'))
            clf.fit(X_train, Y_train)
            accuracy = clf.score(X_test, Y_test)

            print(
                f"The SVM classification accuracy with {kernel} Kernel with {noise}% label noise is: {accuracy * 100:0.2f}%")
            with open('./results/problem_1_SVM.txt', "a") as file:
                file.write(
                    f"The SVM classification accuracy with Gaussian Kernel with {noise}% label noise is: {accuracy * 100:0.2f}%\n")
