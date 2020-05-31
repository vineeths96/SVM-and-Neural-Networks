from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from problem_5.load_data import load_data


# SVM Classifier
def problem_5_SVM():
    os.remove('./results/problem_5_SVM.txt')

    # Load data
    X_train, Y_train, X_test, Y_test = load_data('/EEG/eeg_data.csv')

    # Gaussian Kernel
    gauss_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale'))
    gauss_clf.fit(X_train, Y_train)
    gauss_accuracy = gauss_clf.score(X_test, Y_test)

    print(
        f"The SVM classification accuracy with Gaussian Kernel is: {gauss_accuracy * 100:0.2f}%")
    with open('./results/problem_5_SVM.txt', "a") as file:
        file.write(
            f"The SVM classification accuracy with Gaussian Kernel is: {gauss_accuracy * 100:0.2f}%\n")

    # Polynomial Kernel
    poly_clf = make_pipeline(StandardScaler(), SVC(kernel='poly', gamma='scale'))
    poly_clf.fit(X_train, Y_train)
    poly_accuracy = poly_clf.score(X_test, Y_test)

    print(
        f"The SVM classification accuracy with Polynomial Kernel is: {poly_accuracy * 100:0.2f}%")
    with open('./results/problem_5_SVM.txt', "a") as file:
        file.write(
            f"The SVM classification accuracy with Polynomial Kernel is: {poly_accuracy * 100:0.2f}%\n")

    # Linear Kernel
    linear_clf = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='scale'))
    linear_clf.fit(X_train, Y_train)
    linear_accuracy = poly_clf.score(X_test, Y_test)

    print(
        f"The SVM classification accuracy with Linear Kernel is: {linear_accuracy * 100:0.2f}%")
    with open('./results/problem_5_SVM.txt', "a") as file:
        file.write(
            f"The SVM classification accuracy with Linear Kernel is: {linear_accuracy * 100:0.2f}%\n")
