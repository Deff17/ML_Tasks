import os
import random

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

points_colors = {
    (237, 28, 36): ("red", 1),
    (0, 162, 232): ("blue", 2),
    (255, 255, 255): ("white", 3)
}

color_points = {
    1: "#f45942",
    2: "#4186f4"
}

x_dec = []
for x_0 in np.linspace(0, 301, 70):
    for x_1 in np.linspace(0, 165, 70):
        x_dec.append([x_0, x_1])
x_dec = np.array(x_dec)


# for more realistic data_knn
def skew_point_with_noice(point: float):
    return point + random.gauss(0, 0.5)


def load_data_from_file(img_name: str):
    data_path = os.path.join(os.getcwd(), 'data_svm')
    image_path = os.path.join(data_path, img_name)
    im = Image.open(image_path)
    pix = im.load()
    x_size, y_size = im.size
    print("Size: " + str(x_size) + ' ' + str(y_size))
    data = []
    for x in range(x_size):
        for y in range(y_size):
            if points_colors.get(pix[x, y])[0] is not 'white':
                data.append([skew_point_with_noice(x), skew_point_with_noice(y), points_colors.get(pix[x, y])[1]])
    data_set = pd.DataFrame(data,
                            columns={"x_coordinate", "y_coordinate", "color"})
    return data_set, x_size, y_size


def test_SVM(kernel: str, Cs: list, data, gamma='auto_deprecated',
             coef0=0.0):
    X = np.array(data.iloc[:, 0:2])
    y = np.array(data.iloc[:, 2:])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    accs_test = []
    widths = []
    accs_trains = []

    for c in Cs:
        svm = SVC(C=c, kernel=kernel, gamma=gamma, coef0=coef0)
        svm.fit(X_train, y_train)

        acc_train = svm.score(X_train, y_train)
        acc_test = svm.score(X_test, y_test)

        dec = svm.decision_function(x_dec)
        width = np.sum(np.abs(dec) < 1) / len(x_dec)

        accs_trains.append(acc_train)
        accs_test.append(acc_test)
        widths.append(width)
        plot_data_grouped(svm, kernel + " with c = " + str(c) + " and gamma =" + str(gamma), X_test, y_test)

    return accs_trains, accs_test, widths


def plot_results(acc_trains: list, acc_test: list, widths: list, x: list):
    plt.figure(figsize=(10, 10))
    plt.title("Accuracy of SWM")
    plt.xlabel("Log C value.")
    plt.ylabel("accuracy")
    plt.plot(x, acc_trains, label="Train data")
    plt.plot(x, acc_test, label="Test data")
    plt.legend()

    plt.figure(figsize=(10, 10))
    plt.title("Margin_width(C)")
    plt.xlabel("Log C value.")
    plt.ylabel("margin width")
    plt.plot(x, widths)
    plt.show()


def plot_data_grouped(svm, title, X_train, y_train):
    decisions = svm.decision_function(x_dec)
    for sample, decision in zip(x_dec, decisions):
        alpha = min(np.abs(decision), 1)
        plt.scatter(sample[0], sample[1], c=["#79a9f7" if decision > 0 else "#efad6b"], alpha=alpha)
    y_train = list(map(lambda x: color_points.get(x), y_train.ravel()))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.title(f"Type: svm {title}")
    plt.show()


if __name__ == "__main__":
    data, x_size, y_size = load_data_from_file("dataSet.png")

    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    x_s = np.log10(Cs)

    accs_trains, accs_test, widths = test_SVM(kernel='linear', Cs=Cs, data=data)
    plot_results(accs_trains, accs_test, widths, x_s)

    accs_trains, accs_test, widths = test_SVM(kernel='poly', Cs=Cs, gamma='scale', coef0=3, data=data)
    plot_results(accs_trains, accs_test, widths, x_s)

    accs_trains, accs_test, widths = test_SVM(kernel="rbf", Cs=Cs, gamma=0.0001, data=data)
    plot_results(accs_trains, accs_test, widths, x_s)

    accs_trains, accs_test, widths = test_SVM(kernel="rbf", Cs=Cs, gamma=0.01, data=data)
    plot_results(accs_trains, accs_test, widths, x_s)

    accs_trains, accs_test, widths = test_SVM(kernel="rbf", Cs=Cs, gamma=1, data=data)
    plot_results(accs_trains, accs_test, widths, x_s)
