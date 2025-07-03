import torch
import torch.nn as nn

class SimpleFCNN(nn.Module):
    """A simple Fully Connected Neural Network."""
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initializes the FCNN.

        Args:
            input_size (int): The number of input features.
            hidden_sizes (list of int): A list of sizes for the hidden layers.
            output_size (int): The number of output units.
        """
        super(SimpleFCNN, self).__init__()
        
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        
        # Create hidden layers with ReLU activation
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            
        # Create the output layer
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the network.
        """
        return self.network(x)

def build_model(config):
    """
    Builds a SimpleFCNN model from the configuration.

    Args:
        config (dict): A dictionary containing model parameters.

    Returns:
        SimpleFCNN: The instantiated model.
    """
    input_size = config['model']['input_size']
    hidden_sizes = config['model']['hidden_sizes']
    output_size = config['model']['output_size']
    
    model = SimpleFCNN(input_size, hidden_sizes, output_size)
    
    return model