import numpy as np

class Network(object):
    def __init__(self, layers, costFunction):
        """ layers is a list containing numbers of neurons
        in the layers, from the first to the last.
        """
        self.layersNumber = len(layers)
        self.layers = layers
        self.initializeLayers()
        self.costFunction = costFunction
        
    def initializeLayers(self):
        """Initialze weights with random numbers from 
        standard normal distribution. 
        """
        self.weights = [ np.random.randn(y, x) 
            for y,x in zip(self.layers[1:], self.layers[:-1]) ]
        print(self.weights)





n = Network([5,6,3,1],1)
