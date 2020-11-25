#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def CostFunction(y_hat, y):
    m = y.shape[0]
    return -1 / m * np.nansum(
        np.matmul(np.log(y_hat), y) + np.matmul(np.log(1 - y_hat), 1 - y))


def Accuracy(y_hat, y):
    y_hat = y_hat > 0.5 + 0
    y = y.reshape(-1, 1)
    return 1-np.mean(np.abs(y_hat-y))


class BigSmallNumberNN():
    def __init__(self, dimensions):
        self.W1 = np.random.randn(dimensions[1], dimensions[0]) * 0.1
        self.b1 = np.zeros((dimensions[1], 1))

        self.W2 = np.random.randn(dimensions[2], dimensions[1]) * 0.1
        self.b2 = np.zeros((dimensions[2], 1))

        self.W3 = np.random.randn(dimensions[3], dimensions[2]) * 0.1
        self.b3 = np.zeros((dimensions[3], 1))

    def forward(self, X):
        self.A0 = X

        self.Z1 = np.matmul(self.W1, self.A0) + self.b1
        self.A1 = self.ReLU(self.Z1)

        self.Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.A2 = self.ReLU(self.Z2)

        self.Z3 = np.matmul(self.W3, self.A2) + self.b3
        self.A3 = self.Sigmoid(self.Z3)
        return self.A3

    def ReLU(self, z):
        return np.maximum(z, 0)

    def Sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, y):
        m = y.shape[0]
        self.dZ3 = 1 / m * (self.A3 - y)
        self.dW3 = np.matmul(self.dZ3, self.A2.T)
        self.db3 = np.sum(self.dZ3, axis=1, keepdims=True)

        self.dA2 = np.matmul(self.W3.T, self.dZ3)
        self.dZ2 = np.multiply(self.dA2, np.int64(self.Z2 > 0))
        self.dW2 = np.matmul(self.dZ2, self.A1.T)
        self.db2 = np.sum(self.dZ2, axis=1, keepdims=True)

        self.dA1 = np.matmul(self.W2.T, self.dZ2)
        self.dZ1 = np.multiply(self.dA1, np.int64(self.Z1 > 0))
        self.dW1 = np.matmul(self.dZ1, self.A0.T)
        self.db1 = np.sum(self.dZ1, axis=1, keepdims=True)

    def update(self, lr=0.001):
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3

        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1


X, y = load_digits(return_X_y=True)
y = (y >= 5) + 0
costs = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
dimensions = [(X_train.T).shape[0], 32, 10, 1]
nn = BigSmallNumberNN(dimensions)
lr, desc = 0.01, False

for i in range(30000):
    nn_out = nn.forward(X_train.T)
    cost = CostFunction(nn_out, y_train)
    costs.append(cost)
    if i % 1000 == 0:
        print(i, cost)
        if cost < 0.001 and not desc:
            lr /= 10
            desc = True
    nn.backward(y_train.T)
    nn.update(lr)

print(Accuracy(nn.forward(X_test.T).T, y_test))
plt.plot(costs)
plt.show()
