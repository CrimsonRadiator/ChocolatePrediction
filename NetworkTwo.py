import numpy as np

class Network(object):

    def __init__( self, inputs_n, hidden_n, output_n ):
        self.inputs_n = inputs_n
        self.hidden_n = hidden_n
        self.output_n = output_n

        self.weights =  [   np.random.randn( hidden_n, inputs_n ),
                            np.random.randn( output_n, hidden_n )  ]

        self.biases =   [   np.random.randn( hidden_n, 1 ),
                            np.random.randn( output_n, 1 )  ]


    def feedforward(self, a):
        a1 = np.dot( a, self.weights[0] ) + self.biases[0] 
        z1 = ReLU( a1 )
        a2 = np.dot( z1, self.weights[1] ) + self.biases[1]
        return a2

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
        a2 = ReLU ( z2 )

        #backward pass
        #cost = 0.5*( ( x-y )**2 )    
        
        nabla_b = [ np.zeros( b.shape ) for b in self.biases ]
        nabla_w = [ np.zeros( w.shape ) for w in self.weights ]
        
        delta = np.add(a2, -y) * ReLUprime( z2 ) # (output , 1) . (output, 1)
        nabla_b[1] = delta
        nabla_w[1] = np.dot( delta, a1.T) # (output, 1) . (1, hidden)
        
        delta = np.dot(self.weights[0], a0) * ReLUprime( z1 ) #(hidden, 1)
        nabla_b[0] = delta
        nabla_w[0] = np.dot(delta, a0.T) #(hidden, 1) . (1, input_n)

        return ( nabla_b, nabla_w )

        
def ReLU(x):
    """ReLU function for hidden layer"""
    return np.maximum(x, 0)

def ReLUprime(x):
    """ReLU derivative"""
    x[ x<=0 ] = 0
    x[ x>0 ] = 1
    return x


n = Network(10, 15, 1)



s = np.random.randn(10,1)
print(n.backprop(s, s.sum()))
