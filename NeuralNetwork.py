import numpy as np

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
        



n = Network([2,4,1],1)
print(n.feedforward([1,0]))
