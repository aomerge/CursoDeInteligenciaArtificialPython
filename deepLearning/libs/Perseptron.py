import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np  # type: ignore

class SimplePerceptron(nn.Module):
    """Simple Perceptron with one hidden layer and ReLU activation function."""
    def __init__(self, input_shape, output_shape, divice = torch.device("cpu")):
        """ 
        @param input_size: Number of input neurons
        @param output_size: Number of output neurons
        @param device: Device to run the model
        """
        super(SimplePerceptron, self).__init__()
        self.device = divice
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.lineal = nn.Linear(self.input_shape, self.hidden_shape)
        self.out = nn.Linear(self.hidden_shape, output_shape) 
        self.relu = nn.ReLU()

        
    def forward(self, x):
        """Forward pass of the neural network"""
        if isinstance(x, tuple):
        # Assuming x is a tuple of NumPy arrays and you need the first element
            x = x[0]
        if not isinstance(x, np.ndarray):
            raise TypeError("Expected np.ndarray, got {}".format(type(x).__name__))
        ## x = torch.tensor(x).float().to(self.device) # Convert to tensor
        x  = torch.from_numpy(x).float().to(self.device) # Convert to tensor
        x = nn.functional.relu(self.lineal(x))
        x = self.out(x)
        return x
            