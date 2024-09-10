# _*_coding:utf-8_*_
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def computeCost(x, y, theta):
    theta = np.matrix(theta)
    X = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))
