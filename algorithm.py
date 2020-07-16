""" OneVsAll to predict handwritten numbers """

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fmin_tnc
from scipy.io import loadmat
import matplotlib.image as mpimg


def sigmoid(x):
    """ Sigmoid function or activation function

    Parameters:
        X (vector): Training examples data

    Retunrs:
        (number): Probabilty of training example
    """
    return 1 / (1 + np.exp(-x))


def hθ(theta, X):
    """ Logistic regretion hypotesis, to describe our data

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data

    Retunrs:
        (vector): Vector of probabilities
    """
    return sigmoid(np.dot(X, theta))


def J(theta, X, y, λ):
    """ Cost function J, convex sigmoid function

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Labels for our data

    Retunrs:
        (number): Convex sigmoid function
    """

    return (
        1/len(y) * (
            np.sum(np.dot(-y, np.log(hθ(theta, X))) -
                   np.dot((1 - y), np.log(1 - hθ(theta, X))))) +
        λ/(2*len(y)) * np.sum(theta[1:]**2))


def derivated_term_J(theta, X, y, λ):
    """ Logistic regression function derivated

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Right answers for our data

    Retunrs:
        (vector): Tentative parameters theta
    """
    regterm = copy.deepcopy(theta) # That was the f*cking trick
    regterm[0] = 0
    θ =  1/len(y) * np.dot(X.T, (hθ(theta, X) - y)) + (λ/len(y) * regterm)
    return θ


def pro_min_functions(theta, X, y, λ):
    """ Python minimizing tool kit <3 """
    return fmin_tnc(
        func=J, x0=theta, fprime=derivated_term_J,
        args=(X, y, λ))[0]


def oneVsAll(X, y, K, λ):
    """ Follow this guy https://github.com/Benlau93 """

    m, n = X.shape[0], X.shape[1]
    initial_theta = np.zeros((n + 1, 1))
    all_theta = []
    X = np.hstack((np.ones((m,1)),X))

    for i in range(1, K + 1):
        theta = pro_min_functions(
            initial_theta, X, np.where(y==i, 1, 0).flatten(), λ)
        all_theta.extend(theta)

    return np.array(all_theta).reshape(K, n + 1)


def predictOneVsAll(all_theta, x):
    """ Follow this guy https://github.com/Benlau93 """

    m = x.shape[0]
    x = np.insert(x, 0, 1)
    predictions = x @ all_theta.T
    return np.argmax(predictions,axis=0) + 1


if __name__ == "__main__":
    data = loadmat("ex3data1.mat")
    X = data["X"]
    y = data["y"]
    K = len(np.unique(y))
    λ = 1

    fig, axis = plt.subplots(10,10,figsize=(8,8))
    axis[0, 0].imshow(X[1258,:].reshape(20,20,order="F"), cmap="hot")
    axis[0, 0].axis("off")
    plt.show()

    θ = oneVsAll(X, y, K, λ)
    prediction = predictOneVsAll(θ, X[1258])
    print("It looks like number: ", prediction)
