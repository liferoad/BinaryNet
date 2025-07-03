import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryLinear(nn.Linear):
    """A linear layer with binary weights (-1 or 1) and straight-through estimator."""
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        # Binarize weights to -1 or 1
        binary_weight = _SignActivationFunction.apply(self.weight)
        
        return F.linear(input, binary_weight, self.bias)

class _SignActivationFunction(torch.autograd.Function):
    """Sign activation function with straight-through estimator for gradients."""
    @staticmethod
    def forward(ctx, input):
        # Apply sign function to get -1, 0, or 1
        output = torch.sign(input)
        # Convert any 0s to 1s to ensure only -1 or 1 outputs
        output = torch.where(output == 0, torch.ones_like(output), output)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Straight-through estimator: gradient is 1 for inputs in [-1, 1], 0 otherwise
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

class SignActivation(nn.Module):
    """Wrapper module for SignActivation function."""
    def forward(self, x):
        return _SignActivationFunction.apply(x)

class BinaryFCNN(nn.Module):
    """A Fully Connected Neural Network with binary weights and activations."""
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initializes the Binary FCNN.

        Args:
            input_size (int): The number of input features.
            hidden_sizes (list of int): A list of sizes for the hidden layers.
            output_size (int): The number of output units.
        """
        super(BinaryFCNN, self).__init__()
        
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        
        # Create hidden layers with Sign activation
        for i in range(len(layer_sizes) - 1):
            layers.append(BinaryLinear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(SignActivation())
            
        # Create the output layer
        layers.append(BinaryLinear(layer_sizes[-1], output_size))
        # Add final activation to ensure outputs are -1 or 1
        layers.append(SignActivation())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the network with values -1 or 1.
        """
        return self.network(x)

def build_model(config):
    """
    Builds a BinaryFCNN model from the configuration.

    Args:
        config (dict): A dictionary containing model parameters.

    Returns:
        BinaryFCNN: The instantiated model.
    """
    input_size = config['model']['input_size']
    hidden_sizes = config['model']['hidden_sizes']
    output_size = config['model']['output_size']
    
    model = BinaryFCNN(input_size, hidden_sizes, output_size)
    
    return model