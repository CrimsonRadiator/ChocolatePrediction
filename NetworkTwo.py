import numpy as np
import random
import math

class Network(object):

    def __init__( self, inputs_n, hidden_n, output_n ):
        
        self.inputs_n = inputs_n
        self.hidden_n = hidden_n
        self.output_n = output_n

        self.weights =  [   np.random.randn( hidden_n, inputs_n ) ,
                            np.random.randn( output_n, hidden_n ) ]

        self.biases =   [   np.random.randn( hidden_n, 1 ),
                            np.random.randn( output_n, 1 )  ]
        print(self.weights, self.biases)

    def backprop(self, x, y):
        """
        x has dims (inputs_n, 1)
        y has dims (output_n, 1)
        """
        #forward pass
        a0 = x
        z1 = np.dot( self.weights[0], x ) + self.biases[0] # (hidden, 1)
        a1 = ReLU( z1 ) # hidden x 1
        z2 = np.dot( self.weights[1], a1 ) + self.biases[1] # (output, 1)
        a2 =  z2

        #backward pass
        #cost = 0.5*( ( x-y )**2 )    
        
        nabla_b = [ np.zeros( b.shape ) for b in self.biases ]
        nabla_w = [ np.zeros( w.shape ) for w in self.weights ]


        delta = (a2-y) 
        nabla_b[1] = delta
        nabla_w[1] = np.dot( delta, a1.T) # (output, 1) . (1, hidden)
        
        delta = np.dot(self.weights[0], a0) * ReLUprime( z1 ) #(hidden, 1)
        nabla_b[0] = delta
        nabla_w[0] = np.dot(delta, a0.T) #(hidden, 1) . (1, input_n)

        return ( nabla_b, nabla_w )

    def SGD(self, trainingData, miniBatchSize, lRate, epochs, testData):
        """Stochastic Gradient Descent implementation.
        trainingData is a list op pairs (x,y)
        where x is an input and y is a desired output.
        """
        n = trainingData.__len__()

        for i in range(epochs):
            print("epoch: ", i)
            # first we have to shuffle our training data for each epoch
            random.shuffle(trainingData)
            # then we split the data into equal-sized minibatches
            batch = [trainingData[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)]

            # then we run SGD for each mini-batch
            for miniBatch in batch:
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                for network_input, desired_output in miniBatch:
                    # by our convention, first element contain inputs, and last element is desired output
                    delta_nabla_b, delta_nabla_w = self.backprop(network_input, desired_output)
                    
                    # assign modifiers calculated in backpropagation to apropriate positions
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
               
                # update our weights by applying results from backpropagation with respect to learning rate and by
                # dividing result by length of minibatch (so we have nice average across whole minibatch)
                self.weights = [w - nw * (lRate/len(miniBatch)) for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - nb * (lRate/len(miniBatch)) for b, nb in zip(self.biases, nabla_b)]
        print("w",self.weights,"b", self.biases) 
    def feedforward(self, x):
        z1 = np.dot(self.weights[0], x) + self.biases[0] # (hidden, 1)
        a1 = ReLU(z1) # hidden x 1
        z2 = np.dot(self.weights[1], a1) + self.biases[1] # (output, 1)
        a2 = z2
        return a2
        
def ReLU(x):
    """ReLU function for hidden layer"""
    #return np.maximum(x, 0)
    return 1.0/(1.0+np.exp(-x))

def ReLUprime(x):
    """ReLU derivative"""
    #x[ x<=0 ] = 0
    #x[ x>0 ] = 1
    #return x
    return ReLU(x)*(1-ReLU(x))
