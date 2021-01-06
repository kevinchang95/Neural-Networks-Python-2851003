import numpy as np
import math

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By defaul it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        
        self.weights = (np.random.rand(inputs+1) * 2) -1
        self.bias = bias
        
    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        
        sum = np.dot(np.append(x,self.bias),self.weights)
        
        return self.sigmoid(sum)
    
    
    def set_weight(self, w_init):
        
        if(w_init.shape != self.weights.shape):
            print("Weights length ERROR!!!")
            return
        
        self.weights = w_init
        
        
    def sigmoid(self, x):
        
        return 1 / (1 + math.exp(-x));
        


p = Perceptron(10)

weights = np.arange(11)

x = np.ones(10)

p.set_weight(weights)

sum = p.run(x)

print(sum)

    
    
    
