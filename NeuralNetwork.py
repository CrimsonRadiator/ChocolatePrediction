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
        self.weights = [ np.random.randn(y, x+1) 
            for y,x in zip(self.layers[1:], self.layers[:-1]) ]

    def feedforward(self, x):
        """Return network output for input x"""
        for w in self.weights:
            #add bias
            x = np.append(x, [[1]], axis=0)
            x = self.sigmoid(np.dot(w,x))
        return x

    def costFunction(self, x, y):
        """Calculate MSE of input x and y"""
        a = feedforward(x)
        return 0.5*np.linalg.norm(a, y)**2


    def backpropagation(self, x, y):
        """Backpropagation algorighm.
        x is numpy.ndarray (shape (?,1)) of inputs.
        y is numpy.ndarray (shape (?,1)) of desired outputs.
        """
        a = [x]
        z = []
        for w in self.weights:
            #add bias
            zw = np.dot(w, np.append(x,[[1]],axis=0))
            x = self.sigmoid(zw)
            z.append(zw)
            a.append(x)

        delta = (a[-1] - y) * self.sigmoidPrime(z[-1])

        nabla = [np.zeros(w.shape) for w in self.weights]

        nabla[-1] = np.dot(delta,np.append(a[-2], [[1]], axis=0).transpose())
        
        for l in range(2, len(self.layers)):
            z = z[-l]
            sp = np.append(self.sigmoidPrime(z), [[1]], axis=0)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla[-l] = np.dot(delta, np.append(a[-l-1],[[1]],axis=0).transpose())
        return  nabla

    def sigmoid(self, z):
        """The sigmoid function"""
        return 1.0/(1.0+np.exp(-z))

    def SGD(self, trainingData, miniBatchSize, lRate, epochs):
        """Stochastic Gradient Descent implementation.
        trainingData is a list op pairs (x,y)
        where x is an input and y is a desired output.
        """
        for i in range(epochs):
            # We create mini batch by taking samples from training data
            miniBatch = random.sample(trainingData, miniBatchSize)

            # prepare table which will held modifiers for weights after backpropagation
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for row in miniBatch:
                #by our convention, first element contain inputs, and last element is desired output
                desired_output = row[-1]
                network_input = row[0]

                # print(nabla_w)
                delta_nabla_w = self.backprop(network_input, desired_output)

                # assign modifiers calculated in backpropagation to apropriate positions
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                # update our weights by applying results from backpropagation with respect to learning rate
                self.weights = [w - (lRate/len(miniBatch)) * nw for w, nw in zip(self.weights, nabla_w)]
        
    def sigmoidPrime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))


n = Network([2,5,1])
X = np.ndarray(shape=(2,1), dtype=float, buffer=np.array([2,4]))
Y = np.ndarray(shape=(1,1), dtype=float, buffer=np.array([1]))
print(n.backpropagation(X,Y))
