import numpy as np


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x: np.array) -> np.array:
    return sigmoid(x) * (1 - sigmoid(x))


def output_error(a: np.array, y: np.array, z: np.array):
    return 2 * sigmoid_deriv(z) * (a - y)


def label_one_hot_encoded(label: int, number_classes: int) -> np.array:
    y = np.zeros(number_classes)
    y[label] = 1
    return y
