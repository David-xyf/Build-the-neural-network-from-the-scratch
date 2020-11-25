# Design

## Abstract

In this report, I will design a neural network by hand which is shown below. Instead of using a framework of NN, I will implement the function of distinguishing the number (5-9) which is bigger than or equals to 5 with the number (0-5) which is smaller than 5. Basing on the sklearn's dataset, I use *load_digits* to load the dataset where each data point is a 8x8 image of a digit, to train NN on the training set and test the accuracy on the test set.

![nn](https://github.com/David-xyf/Build-the-neural-network-from-the-scratch/raw/main/images/nn.png)

## Mathematic derivation

1. **Initialize the weight and bias**. In order to make sure the shape of matric, I make a table below.

   | Shape | Input Layer | Hidden Layer1 | Hidden Layer2 | Output Layer |
   | ----- | ----------- | ------------- | ------------- | ------------ |
   | X     | (64, m)     | (32, m)       | (10, m)       | (1, m)       |
   | W     | (32, 64)    | (10, 32)      | (1, 10)       |              |
   | b     | (32, 1)     | (10, 1)       | (1, 1)        |              |

   I squeeze the 8*8 images of digits and then use 80% images to train 3-layer network, 20% to test it.

2. After setting the weights and bias, we will come into the duration of **the forward propagation**. During this period, we can use ReLU and sigmoid as the activation functions in neurons. The math flowchart is below:
   $$
   z^{[1]}=w^{[1]}x+b^{[1]},a^{[1]}=ReLU(z^{[1]})\\
   z^{[2]}=w^{[2]}a^{[1]}+b^{[2]},a^{[2]}=ReLU(z^{[2]})\\
   z^{[3]}=w^{[3]}a^{[2]}+b^{[3]},a^{[3]}=sigmoid(z^{[3]})
   $$

3. The final step is the most important duration, which is called "**Backward Propagation**". In this step, I will derive gradients for all the network parameters and introduce the L2 norm to improve the NN. So I get the cost function with L2:
   $$
   J=-\frac{1}{m}\sum_{i=0}^m [y^{(i)}log(\hat y^{(i)})+(1-y^{(i)})log(1- \hat y^{(i)})]+ \frac {\lambda}{2m}\sum_j w^2_j
   $$
   According to the cost function, I can derive the gradients for $w^{i}$ and $b^i$ (ignore the L2 regularization):
   
   
   $$
   \frac{\partial J}{\partial w^{[3]}}=\frac{\partial J}{\partial a^{[3]}}\frac{\partial a^{[3]}}{\partial z^{[3]}}\frac{\partial z^{[3]}}{\partial w^{[3]}}\\
   \frac{\partial J}{\partial a^{[3]}}=\frac{\partial J}{\partial \hat y^{[3]}}=-\frac{1}{m}\frac{y-a^{[3]}}{a^{[3]}(1-a^{[3]})}\\
   \frac{\partial a^{[3]}}{\partial z^{[3]}}=a^{[3]}(1-a^{[3]})\\
   \frac{\partial z^{[3]}}{\partial w^{[3]}}=a^{[2]}\\
   $$
   So, after simplifying, the gradients are:
   $$
   dw^{[3]}=\frac{\partial J}{\partial w^{[3]}}=\frac{1}{m}(a^{[3]}-y)a^{[2]}=dz^{[3]}*a^{[2]}\\
   db^{[3]}=\frac{1}{m}(a^{[3]}-y)=dz^{[3]}
   $$
   Basing on the chain rule, we can easily get general formulas to calculate all gradients on every layer:
   $$
   da^{[l]}=w^{[l+1]}dz^{[l+1]}\\
   dz^{[l]}=da^{[l]}g'(z^{[l]})\\
   dw^{[l]}=dz^{[l]}a^{[l-1]}\\
db^{[l]}=dz^{[l]}
   $$
   
4. After training weights based on gradient descent, we can get NN to **clarify the number which is bigger or less than 5**. If the number is less than 5, the output will be nearly 0. Otherwise, that will be closed to 1.   

# Implementation

## Introduction

In order to get NN, I use python to program the class of NN and common functions, such as cost function. I write 3 versions of NN from easy network which is fixed the layers' dimensions to complex network which can be changed the layers' depth and dimension flexibly.

## Code Analysis

### Neural Network with fixed layers and dimensions

Import the python libraries needed

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
```

Definition the basic cost function

```python
def CostFunction(y_hat, y):
    """ To calculate the cost from y_hat and y """
    m = y.shape[0]
    return -1 / m * np.nansum(
        np.matmul(np.log(y_hat), y) + np.matmul(np.log(1 - y_hat), 1 - y))
```

Calculate the accuracy of outputs

```python
def Accuracy(y_hat, y):
    """ According the y_hat and y, to calculate the accuracy of neural network """
    y_hat = y_hat > 0.5 + 0
    y = y.reshape(-1, 1)
    return 1-np.mean(np.abs(y_hat-y))
```

Design the neural network for fixed dimensions and layers

```python
class BigSmallNumberNN():
    def __init__(self):
        self.W1 = np.random.randn(32, 64) * 0.1
        self.b1 = np.zeros((32, 1))

        self.W2 = np.random.randn(10, 32) * 0.1
        self.b2 = np.zeros((10, 1))

        self.W3 = np.random.randn(1, 10) * 0.1
        self.b3 = np.zeros((1, 1))

    def ReLU(self, z):
        """ Activation function: ReLU """
        return np.maximum(z, 0)

    def Sigmoid(self, z):
        """ Activation function: Sigmoid """
        return 1 / (1 + np.exp(-z))


    def forward(self, X):
        """ To achieve the forward propagation """
        self.A0 = X

        self.Z1 = np.matmul(self.W1, self.A0) + self.b1
        self.A1 = self.ReLU(self.Z1)

        self.Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.A2 = self.ReLU(self.Z2)

        self.Z3 = np.matmul(self.W3, self.A2) + self.b3
        self.A3 = self.Sigmoid(self.Z3)
        return self.A3

    def backward(self, y):
        """ To achieve the backward propagation """
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
        """ To make gradient descent"""
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3

        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
```

The main function to split the datasets, train NN, and plot the figures of accuracies and costs.

```python
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

def main():
    nn = BigSmallNumberNN()
    X, y = load_digits(return_X_y=True)
    y = (y >= 5) + 0
    costs,accuracies = [],[]
    lr, desc = 0.01, False
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("iterations  Costs")
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
        accuracies.append(Accuracy(nn.forward(X_test.T).T, y_test))
    
    print("Final Accuracy")
    print(Accuracy(nn.forward(X_test.T).T, y_test))
    drawAccuraciesFigure(accuracies)
    drawCostsFigure(costs)

if __name__ == '__main__':
    main()
```

#### Results

![accuracies-iterations-fixed](https://github.com/David-xyf/Build-the-neural-network-from-the-scratch/raw/main/images/accuracies-iterations-fixed.png)

![costs-iterations-fixed](https://github.com/David-xyf/Build-the-neural-network-from-the-scratch/raw/main/images/costs-iterations-fixed.png)

![accuracies-costs-fixed](https://github.com/David-xyf/Build-the-neural-network-from-the-scratch/raw/main/images/accuracies-costs-fixed.png)

#### Pros and Cons

Pros:

1. The code is easy to read and the structure is distinct.
2. The code shows the 3 steps of designing the neural network and justify my math derivation.

Cons:

1. The layers and dimension is fixed, and I can not change the number of layers or dimensions.
2. In a class, I used too many built-in variables and these will occupy much memory.   

# Improvement

### Neural Network with flexible layers and dimensions

After improving slightly, I could make basic NN with flexible dimensions blow. I uses 3 dictionaries in Python to store parameters of initialization, forward propagation and backward propagation.

```python
class BigSmallNumberNN():
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = {}
        for i in range(1, len(layer_dims)):
            self.parameters["W"+str(i)] = np.random.randn(layer_dims[i],
                                                          layer_dims[i-1]) * 0.1
            self.parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))

        self.layers_forward = {}
        self.layers_backward = {}

    def forward(self, X):
        self.layers_forward["A0"] = X

        for i in range(1, len(self.layer_dims)-1):
            self.layers_forward["Z"+str(i)] = np.matmul(self.parameters["W"+str(
                i)], self.layers_forward["A"+str(i-1)]) + self.parameters["b"+str(i)]
            self.layers_forward["A" +
                                str(i)] = self.ReLU(self.layers_forward["Z"+str(i)])

        outputLayerNumber = len(self.layer_dims)-1
        self.layers_forward["Z"+str(outputLayerNumber)] = np.matmul(self.parameters["W"+str(outputLayerNumber)],
                                                                    self.layers_forward["A"+str(outputLayerNumber-1)]) + self.parameters["b"+str(outputLayerNumber)]
        self.layers_forward["A"+str(outputLayerNumber)] = self.Sigmoid(
            self.layers_forward["Z"+str(outputLayerNumber)])
        return self.layers_forward["A"+str(outputLayerNumber)]

    def ReLU(self, z):
        return np.maximum(z, 0)

    def Sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, y):
        m = y.shape[0]
        self.layers_backward["dZ"+str(len(self.layer_dims)-1)] = 1 / m * (
            self.layers_forward["A"+str(len(self.layer_dims)-1)] - y)
        for i in range(len(self.layer_dims)-1, 0, -1):
            if i == len(self.layer_dims)-1:
                self.layers_backward["dW"+str(i)] = np.matmul(
                    self.layers_backward["dZ"+str(i)], self.layers_forward["A"+str(i-1)].T)
                self.layers_backward["db"+str(i)] = np.sum(
                    self.layers_backward["dZ"+str(i)], axis=1, keepdims=True)
            else:
                self.layers_backward["dA"+str(i)] = np.matmul(
                    self.parameters["W"+str(i+1)].T, self.layers_backward["dZ"+str(i+1)])
                self.layers_backward["dZ"+str(i)] = np.multiply(
                    self.layers_backward["dA"+str(i)], np.int64(self.layers_forward["Z"+str(i)] > 0))
                self.layers_backward["dW"+str(i)] = np.matmul(
                    self.layers_backward["dZ"+str(i)], self.layers_forward["A"+str(i-1)].T)
                self.layers_backward["db"+str(i)] = np.sum(
                    self.layers_backward["dZ"+str(i)], axis=1, keepdims=True)

    def update(self, lr=0.001):
        for i in range(1, len(layer_dims)):
            self.parameters["W"+str(i)] -= lr*self.layers_backward["dW"+str(i)]
            self.parameters["b"+str(i)] -= lr*self.layers_backward["db"+str(i)]
```

#### Pros and Cons

Pros:

1. In a class, I used less built-in variables.  
2. The layers and dimensions are flexible to regulate in the neurons.

Cons:

1. The code is a little difficult to read and the structure is not distinct.
2. If we directly program this kind of neural network, it's hard to complete it because this kind of code styles uses much for-loop and needs highly understanding for matrix manipulations between layers.

### Improved Neural Network with L2 norm

Basing on above neural network, I introduce L2 norm to avoid the risk of overfitting.

Firstly, I modify the cost function to cost function with L2 norm.

```python
def ComputeCostWithL2(y_hat, y, parameters, lambd):
    """ To calculate the cost under the influence of weights based on original cost function """
    m = y.shape[0]
    crossEntropyCost = CostFunction(y_hat, y)
    L = len(parameters)//2
    L2RegularizationCost = 0
    for i in range(L):
        L2RegularizationCost += (lambd/(2*m)) * \
            np.sum(np.square(parameters["W"+str(i+1)]))
    return crossEntropyCost+L2RegularizationCost
```

Secondly, I modify the backward propagation in the class with L2 regularization.

```python
class BigSmallNumberNN():
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = {}
        for i in range(1, len(layer_dims)):
            self.parameters["W"+str(i)] = np.random.randn(layer_dims[i],
                                                          layer_dims[i-1]) * 0.1
            self.parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))

        self.layers_forward = {}
        self.layers_backward = {}

    def forward(self, X):
        self.layers_forward["A0"] = X

        for i in range(1, len(self.layer_dims)-1):
            self.layers_forward["Z"+str(i)] = np.matmul(self.parameters["W"+str(
                i)], self.layers_forward["A"+str(i-1)]) + self.parameters["b"+str(i)]
            self.layers_forward["A" +
                                str(i)] = self.ReLU(self.layers_forward["Z"+str(i)])

        outputLayerNumber = len(self.layer_dims)-1
        self.layers_forward["Z"+str(outputLayerNumber)] = np.matmul(self.parameters["W"+str(outputLayerNumber)],
                                                                    self.layers_forward["A"+str(outputLayerNumber-1)]) + self.parameters["b"+str(outputLayerNumber)]
        self.layers_forward["A"+str(outputLayerNumber)] = self.Sigmoid(
            self.layers_forward["Z"+str(outputLayerNumber)])
        return self.layers_forward["A"+str(outputLayerNumber)]

    def ReLU(self, z):
        return np.maximum(z, 0)

    def Sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def backwardWithL2(self, y, lambd):
        m = y.shape[0]
        self.layers_backward["dZ"+str(len(self.layer_dims)-1)] = 1 / m * (
            self.layers_forward["A"+str(len(self.layer_dims)-1)] - y)
        for i in range(len(self.layer_dims)-1, 0, -1):
            if i == len(self.layer_dims)-1:
                self.layers_backward["dW"+str(i)] = np.matmul(self.layers_backward["dZ"+str(
                    i)], self.layers_forward["A"+str(i-1)].T) + lambd/m * self.parameters["W"+str(i)]
                self.layers_backward["db"+str(i)] = np.sum(
                    self.layers_backward["dZ"+str(i)], axis=1, keepdims=True)
            else:
                self.layers_backward["dA"+str(i)] = np.matmul(
                    self.parameters["W"+str(i+1)].T, self.layers_backward["dZ"+str(i+1)])
                self.layers_backward["dZ"+str(i)] = np.multiply(
                    self.layers_backward["dA"+str(i)], np.int64(self.layers_forward["Z"+str(i)] > 0))
                self.layers_backward["dW"+str(i)] = np.matmul(
                    self.layers_backward["dZ"+str(i)], self.layers_forward["A"+str(i-1)].T) + lambd/m * self.parameters["W"+str(i)]
                self.layers_backward["db"+str(i)] = np.sum(
                    self.layers_backward["dZ"+str(i)], axis=1, keepdims=True)

    def update(self, lr=0.001):
        for i in range(1, len(self.layer_dims)):
            self.parameters["W"+str(i)] -= lr*self.layers_backward["dW"+str(i)]
            self.parameters["b"+str(i)] -= lr*self.layers_backward["db"+str(i)]
```

Finally, update the main function and then print the results.

```python
def main():
    X, y = load_digits(return_X_y=True)
    y = (y >= 5) + 0
    costs,accuracies = [],[]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    layer_dims = [(X_train.T).shape[0], 32, 10, 1]
    nn = BigSmallNumberNN(layer_dims)
    lr, desc, lambd = 0.01, False, 0.7
    
    print("iterations  Costs")
    for i in range(30000):
        nn_out = nn.forward(X_train.T)
        cost = ComputeCostWithL2(nn_out, y_train, nn.parameters, lambd)
        costs.append(cost)
        if i % 1000 == 0:
            print(i, cost)
            if cost < 0.001 and not desc:
                lr /= 10
                desc = True
        nn.backwardWithL2(y_train.T, lambd)
        nn.update(lr)
        accuracies.append(Accuracy(nn.forward(X_test.T).T, y_test))
    print("Final Accuracy")
    print(Accuracy(nn.forward(X_test.T).T, y_test))
    drawAccuraciesFigure(accuracies)
    drawCostsFigure(costs)
```

#### Results

![accuracies-iterations-L2](https://github.com/David-xyf/Build-the-neural-network-from-the-scratch/raw/main/images/accuracies-iterations-L2.png)

![costs-iterations-L2](https://github.com/David-xyf/Build-the-neural-network-from-the-scratch/raw/main/images/costs-iterations-L2.png)

![accuracies-costs-L2](https://github.com/David-xyf/Build-the-neural-network-from-the-scratch/raw/main/images/accuracies-costs-L2.png)

#### Pros and Cons

Pros:

1. Avoid the risk of overfitting. But, actually, in this dataset, the effects of overfitting is not obvious even though using many layers and higher dimensions.

Cons:

1. The code is a little difficult to read and the structure is not distinct.
2. It's harder to complete it than improved NN.

# Summary

Through finishing this assignment, I find the magic of mathematics and neural network. Although the neural network still has some shortages, it increases my understanding for NN. I used to program the NN by pytorch, but I cannot fully get the idea of gradient descent. After designing NN by hands, it contributes me to memorize the NN structure and comprehend the NN kernels.
