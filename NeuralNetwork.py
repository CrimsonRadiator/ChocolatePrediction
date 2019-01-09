import numpy as np
import random

class Network(object):
    def __init__(self, layers, costFunction):
        """ layers is a list containing numbers of neurons
        in the layers, from the first to the last.
        """
        self.layers = layers
        self.costFunction = costFunction
        
        """Initialze weights with random numbers from 
        standard normal distribution. +1 for the bias weights. 
        """
        self.weights = [ np.random.randn(y, x+1) 
            for y,x in zip(self.layers[1:], self.layers[:-1]) ]
        print(self.weights)

    """Return network output for input x"""
    def feedforward(self, x):
        for w in self.weights:
            #add bias
            x = np.append(x, 1)
            x = self.sigmoid(np.dot(w,x))
        return x

        """The sigmoid function"""
    def sigmoid(self, z):
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
        



n = Network([2,4,1],1)
data = [[np.random.rand(1, 2), 1] for x in range(300)]
n.SGD(data, 4, 1, 3)
print(n.feedforward([1,0]))
