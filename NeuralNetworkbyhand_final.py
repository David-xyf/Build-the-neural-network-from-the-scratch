# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:23:26 2020

@author: xyf
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def CostFunction(y_hat, y):
    m = y.shape[0]
    return -1 / m * np.nansum(
        np.matmul(np.log(y_hat), y) + np.matmul(np.log(1 - y_hat), 1 - y))


def Accuracy(y_hat, y):
    y_hat = y_hat > 0.5 + 0
    y = y.reshape(-1, 1)
    return 1-np.mean(np.abs(y_hat-y))


class BigSmallNumberNN():
    def __init__(self, Lnums):
        self.Lnums = Lnums
        self.p = {}
        for i in range(1, len(Lnums)):
            self.p["W"+str(i)] = np.random.randn(Lnums[i], Lnums[i-1]) * 0.1
            self.p["b"+str(i)] = np.zeros((Lnums[i], 1))

        self.fp = {}
        self.bp = {}

    def forward(self, X):
        self.fp["A0"] = X

        for i in range(1, len(self.Lnums)-1):
            self.fp["Z"+str(i)] = np.matmul(self.p["W"+str(
                i)], self.fp["A"+str(i-1)]) + self.p["b"+str(i)]
            self.fp["A" +
                    str(i)] = self.ReLU(self.fp["Z"+str(i)])

        Lnum = len(self.Lnums)-1
        self.fp["Z"+str(Lnum)] = np.matmul(self.p["W"+str(Lnum)],
                                           self.fp["A"+str(Lnum-1)]) + self.p["b"+str(Lnum)]
        self.fp["A"+str(Lnum)] = self.Sigmoid(self.fp["Z"+str(Lnum)])
        return self.fp["A"+str(Lnum)]

    def ReLU(self, z):
        return np.maximum(z, 0)

    def Sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, y):
        m = y.shape[0]
        self.bp["dZ"+str(len(self.Lnums)-1)] = 1 / m * (
            self.fp["A"+str(len(self.Lnums)-1)] - y)
        for i in range(len(self.Lnums)-1, 0, -1):
            if i == len(self.Lnums)-1:
                self.bp["dW"+str(i)] = np.matmul(
                    self.bp["dZ"+str(i)], self.fp["A"+str(i-1)].T)
                self.bp["db"+str(i)] = np.sum(
                    self.bp["dZ"+str(i)], axis=1, keepdims=True)
            else:
                self.bp["dA"+str(i)] = np.matmul(
                    self.p["W"+str(i+1)].T, self.bp["dZ"+str(i+1)])
                self.bp["dZ"+str(i)] = np.multiply(
                    self.bp["dA"+str(i)], np.int64(self.fp["Z"+str(i)] > 0))
                self.bp["dW"+str(i)] = np.matmul(
                    self.bp["dZ"+str(i)], self.fp["A"+str(i-1)].T)
                self.bp["db"+str(i)] = np.sum(
                    self.bp["dZ"+str(i)], axis=1, keepdims=True)

    def update(self, lr=0.001):
        for i in range(1, len(Lnums)):
            self.p["W"+str(i)] -= lr*self.bp["dW"+str(i)]
            self.p["b"+str(i)] -= lr*self.bp["db"+str(i)]


def drawAccuraciesFigure(accuracies):
    plt.figure()
    plt.plot(accuracies)
    plt.title("accuracies-iterations")
    plt.xlabel("iterations")
    plt.ylabel("accuracies")
    plt.show()


def drawCostsFigure(costs):
    plt.figure()
    plt.plot(costs)
    plt.title("costs-iterations")
    plt.xlabel("iterations")
    plt.ylabel("costs")
    plt.show()


X, y = load_digits(return_X_y=True)
y = (y >= 5) + 0
costs, accuracies = [], []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
Lnums = [(X_train.T).shape[0], 100, 200, 10, 1]
nn = BigSmallNumberNN(Lnums)
lr, desc = 0.01, False

for i in range(10000):
    nn_out = nn.forward(X_train.T)
    cost = CostFunction(nn_out, y_train)
    costs.append(cost)
    if i % 1000 == 0:
        print(i, cost)
        if cost < 0.001 and not desc:
            lr /= 10
            desc = True
    nn.backward(y_train.T)
    accuracies.append(Accuracy(nn.forward(X_test.T).T, y_test))
    nn.update(lr)

print(Accuracy(nn.forward(X_test.T).T, y_test))

drawAccuraciesFigure(accuracies)
drawCostsFigure(costs)
