import numpy as np
import random


class Network(object):
    def __init__(self, layers):
        """ layers is a list containing numbers of neurons
        in the layers, from the first to the last.
        """
        self.layers = layers
        """Initialze weights with random numbers from 
        standard normal distribution. +1 for the bias weights. 
        """
        self.weights = [np.random.randn(y, x+1)
            for y, x in zip(self.layers[1:], self.layers[:-1])]

    def feedforward(self, x):
        """Return network output for input x"""
        for w in self.weights:
            #add bias
            #print('\nx: ' + x.__str__() + 'w: ' + w.__str__())
            x = np.append(x, [[1]], axis=0)
            x = self.sigmoid(np.dot(w,x))
        return x

    def costFunction(self, x, y):
        """Calculate MSE of input x and y"""
        a = self.feedforward(x)
        return 0.5*(a - y)**2

    def backpropagation(self, x, y):
        """Backpropagation algorighm.
        x is numpy.ndarray (shape (?,1)) of inputs.
        y is numpy.ndarray (shape (?,1)) of desired outputs.
        """
        activations = [x]
        z = []
        for w in self.weights:
            #add bias
            zw = np.dot(w, np.append(x, [[1]], axis=0))
            x = self.sigmoid(zw)
            z.append(zw)
            activations.append(x)

        delta = (activations[-1] - y.transpose()) * self.sigmoidPrime(z[-1])
        nabla = [np.zeros(w.shape) for w in self.weights]
        nabla[-1] = np.dot(delta, np.append(activations[-2], [[1]], axis=0).transpose())

        for i in range(2, len(self.layers)):
            z = z[-i]
            sp = np.append(self.sigmoidPrime(z), [[1]], axis=0)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla[-i] = np.dot(delta, np.append(activations[-i-1], [[1]], axis=0).transpose())
        return nabla

    def SGD(self, trainingData, miniBatchSize, lRate, epochs, testData):
        """Stochastic Gradient Descent implementation.
        trainingData is a list op pairs (x,y)
        where x is an input and y is a desired output.
        """
        n = trainingData.__len__()
        for i in range(epochs):
            tmp = 0
            random.shuffle(trainingData)
            miniBatch = [trainingData[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)]

            # prepare table which will held modifiers for weights after backpropagation
            for row in miniBatch:
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                for network_input, desired_output in row:
                    # by our convention, first element contain inputs, and last element is desired output
                    delta_nabla_w = self.backpropagation(network_input, desired_output)
                    delta_nabla_w[0] = np.delete(delta_nabla_w[0], -1, 0)

                    # delta_nabla_w[1] = np.delete(delta_nabla_w[1],-1)
                    # assign modifiers calculated in backpropagation to apropriate positions
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                # update our weights by applying results from backpropagation with respect to learning rate
                self.weights = [w - (lRate/len(miniBatch)) * nw for w, nw in zip(self.weights, nabla_w)]
            for row in testData:
                if ((row[1] - self.feedforward(row[0]))) > 0:
                    tmp += (row[1] - self.feedforward(row[0])) * 977.0
                else:
                    tmp -= (row[1] - self.feedforward(row[0])) * 977.0
            print('i', i, 'mean: ', tmp / testData.__len__())

    def sigmoid(self, z):
        """The sigmoid function"""
        return 1.0/(1.0+np.exp(-z))

    def sigmoidPrime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))


# n = Network([3, 3, 1])
# X = np.array([[1, 0.5, 0]]).T
# Y = np.array([[0.95]])
# X1 = np.array([[0, 1, 0]]).T
# Y1 = np.array([[0.32]])

# print('x: ' + X.__str__())
# print('y: ' + Y.__str__())
# for i in range(1, 10000):
    # print('weights: ' + n.weights.__str__())
    # n.SGD([[X, Y]], 1, 1, 1)
    # n.SGD([[X1, Y1]], 1, 1, 1)

# print('result: ', n.feedforward(X))
# print('result1: ', n.feedforward(X1))
# print('result1: ', n.feedforward(X2))
