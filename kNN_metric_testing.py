import random

import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from imblearn.under_sampling import CondensedNearestNeighbour

points_colors = {
    (34, 177, 76): ("green", 1),
    (255, 242, 0): ("yellow", 2),
    (237, 28, 36): ("red", 3),
    (0, 162, 232): ("blue", 4),
    (255, 255, 255): ("white", 5)
}

boarders_colors = {
    "light_green": (178, 255, 178),
    "light_yellow": (245, 255, 204),
    "light_red": (255, 216, 204),
    "light_blue": (204, 243, 255)
}

b_colors = ListedColormap(["#b2ffb2", "#f5ffcc", "#ffd8cc", "#ccf3ff"])
p_colors = ListedColormap(["green", "yellow", "red", "blue"])


# for more realistic data_knn
def skew_point_with_noice(point: float):
    return point + random.gauss(0, 0.5)


def load_data_from_file(img_name: str):
    data_path = os.path.join(os.getcwd(), 'data_knn')
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


def plot_decisions_boundaries(X, y, clf=None):
    h = 0.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=b_colors)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=p_colors,
                edgecolor="k", s=10)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def test_knn(data_set: pd.DataFrame, metric: str, k: int, weights='uniform'):
    X = np.array(data_set.iloc[:, 0:2])
    y = np.array(data_set.iloc[:, 2:])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    if metric is "euclidean":
        clf = neighbors.KNeighborsClassifier(k, metric=metric, weights=weights)
    else:
        clf = neighbors.KNeighborsClassifier(k, metric=metric, weights=weights,
                                             metric_params={'V': np.cov(X_train, rowvar=False)})
    clf.fit(X_train, y_train.ravel())
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(predicted, y_test)
    print(accuracy)
    plot_decisions_boundaries(X_train, y_train, clf=clf)


def get_sliced_data(x_coordinate, y_coordinate, step_x, step_y, X_train, y_train, X_test, y_test):
    min_x_train = x_coordinate - 2 * step_x
    max_x_train = x_coordinate + step_x + 1
    min_y_train = y_coordinate - 2 * step_y
    max_y_train = y_coordinate + step_y

    min_x_test = x_coordinate - step_x
    min_y_test = y_coordinate - step_y
    max_x_test = x_coordinate
    max_y_test = y_coordinate

    X_test_result = []
    y_test_result = []
    X_train_result = []
    y_train_result = []
    for i in range(X_train.__len__()):
        if min_x_train < X_train[i][0] < max_x_train and min_y_train < X_train[i][1] < max_y_train:
            X_train_result.append(X_train[i]), y_train_result.append(y_train[i])
    for i in range(X_test.__len__()):
        if min_x_test < X_test[i][0] < max_x_test and min_y_test < X_test[i][1] < max_y_test:
            X_test_result.append(X_test[i]), y_test_result.append(y_test[i])
    return X_train_result, y_train_result, X_test_result, y_test_result


def predict_for_plot(clf: neighbors.KNeighborsClassifier, x_coordinate, y_coordinate, step_x, step_y, x_size, y_size):
    x_min = x_coordinate - 2 * step_x
    x_max = x_coordinate + step_x + 1
    y_min = y_coordinate - 2 * step_y
    y_max = y_coordinate + step_y
    h = 0.2
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > x_size:
        x_max = x_size
    if y_max > y_size:
        y_max = y_size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=b_colors)
    plt.show()
    return Z


def plot_decisions_boundaries_from_slices(Z_appended, x_max, y_max):
    h = 0.2
    x_min, y_min = 0, 0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = Z_appended.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=b_colors)
    # plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=p_colors,
    #             edgecolor="k", s=10)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    plt.show()


def cnn_test(data_set: pd.DataFrame, metric: str, k: int, weights='uniform'):
    X = np.array(data_set.iloc[:, 0:2])
    y = np.array(data_set.iloc[:, 2:])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    cnn = CondensedNearestNeighbour(n_neighbors=k, sampling_strategy="all")
    X_train_re, y_train_re = cnn.fit_resample(X_train, y_train)
    clf = neighbors.KNeighborsClassifier(k, metric=metric, weights=weights)
    clf.fit(X_train_re, y_train_re.ravel())
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(predicted, y_test)
    print(accuracy)
    plot_decisions_boundaries(X_train, y_train, clf=clf)


def test_knn_with_splits(data_set: pd.DataFrame, x_size, y_size):
    side_pieces = 5
    piece_height, piece_width = x_size / side_pieces, y_size / side_pieces
    X = np.array(data_set.iloc[:, 0:2])
    y = np.array(data_set.iloc[:, 2:])
    accuracy_list = []
    Z_appended = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    for i in range(1, side_pieces + 1):
        for j in range(1, side_pieces + 1):
            X_train_sliced, y_train_sliced, X_test_sliced, y_test_sliced = get_sliced_data(i * piece_height,
                                                                                           j * piece_width,
                                                                                           piece_height, piece_width,
                                                                                           X_train, y_train, X_test,
                                                                                           y_test)

            if X_train_sliced is not []:
                clf = neighbors.KNeighborsClassifier(1, metric='mahalanobis', weights='uniform',
                                                     metric_params={'V': np.cov(X_train_sliced, rowvar=False)})
                clf.fit(X_train_sliced, np.array(y_train_sliced).ravel())
                predicted = clf.predict(X_test_sliced)
                accuracy = accuracy_score(y_test_sliced, predicted)
                accuracy_list.append(accuracy)
                print("Accuracy: " + str(accuracy))
                Z = predict_for_plot(clf, i * piece_height, j * piece_width, piece_height, piece_width, x_size, y_size)
    #             if Z_appended is None:
    #                 Z_appended = Z
    #             else:
    #                 Z_appended = np.concatenate([Z_appended, Z])
    #             print(Z_appended.shape, Z_appended.size)
    # plot_decisions_boundaries_from_slices(Z_appended, (side_pieces + 1) * piece_height, (side_pieces + 1) * piece_width)
    print("Average acc: " + str(np.mean(accuracy_list)))


if __name__ == '__main__':
    pics = ["firstDataSet.png", "secondDataSet.png", "thirdDataSet.png"]
    # load_data_from_file(pics[0])
    # load_data_from_file(pics[1])
    # load_data_from_file(pics[2])
    # test_knn(load_data_from_file(pics[1]), 'mahalanobis', 1)
    # test_knn(load_data_from_file(pics[1]), "euclidean", 1)
    # test_knn(load_data_from_file(pics[1]), "euclidean", 3)
    # test_knn(load_data_from_file(pics[1]), "euclidean", 1, weights='distance')
    for pic in pics:
        data, x_size, y_size = load_data_from_file(pic)
        cnn_test(data, "euclidean", 3)

    # data_knn, x_size, y_size = load_data_from_file(pics[0])
    # # test_knn_with_splits(data_knn, x_size, y_size)
    # cnn_test(data_knn,"euclidean", 1)
    # cnn_test(data_knn, "euclidean", 3)
