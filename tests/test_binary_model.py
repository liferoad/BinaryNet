import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bnet.models.binary_fcnn import BinaryFCNN, BinaryLinear, SignActivation

def test_binary_linear():
    """Test that BinaryLinear layer uses binary weights."""
    print("\nTesting BinaryLinear layer...")
    layer = BinaryLinear(5, 3)
    
    # Initialize weights with random values
    nn_init_weights = torch.randn(3, 5)
    layer.weight.data = nn_init_weights
    
    # Create a random input
    x = torch.randn(2, 5)
    
    # Forward pass
    output = layer(x)
    
    # Get the actual weights used in the forward pass
    binary_weight = torch.sign(layer.weight)
    binary_weight = torch.where(binary_weight == 0, torch.ones_like(binary_weight), binary_weight)
    
    # Verify weights are binary (-1 or 1)
    unique_weights = torch.unique(binary_weight)
    print(f"Unique weight values: {unique_weights.detach().numpy()}")
    assert len(unique_weights) <= 2, "Weights should only be -1 or 1"
    assert torch.all(torch.abs(binary_weight) == 1), "All weights should have magnitude 1"
    
    # Verify the output matches manual calculation with binary weights
    manual_output = torch.matmul(x, binary_weight.t())
    if layer.bias is not None:
        manual_output += layer.bias
    
    assert torch.allclose(output, manual_output), "Output doesn't match expected calculation"
    print("BinaryLinear test passed!")

def test_sign_activation():
    """Test that SignActivation outputs only -1 or 1."""
    print("\nTesting SignActivation...")
    activation = SignActivation()
    
    # Test with various inputs including zeros
    inputs = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    outputs = activation(inputs)
    
    print(f"Inputs: {inputs.detach().numpy()}")
    print(f"Outputs: {outputs.detach().numpy()}")
    
    # Verify outputs are only -1 or 1
    unique_outputs = torch.unique(outputs)
    print(f"Unique output values: {unique_outputs.detach().numpy()}")
    assert len(unique_outputs) <= 2, "Outputs should only be -1 or 1"
    assert torch.all(torch.abs(outputs) == 1), "All outputs should have magnitude 1"
    print("SignActivation test passed!")

def test_binary_fcnn():
    """Test the full BinaryFCNN model."""
    print("\nTesting BinaryFCNN model...")
    model = BinaryFCNN(input_size=5, hidden_sizes=[4, 3], output_size=2)
    
    # Create a random input
    x = torch.randn(2, 5)
    
    # Forward pass
    output = model(x)
    
    print(f"Model output shape: {output.shape}")
    print(f"Model output values: {output.detach().numpy()}")
    
    # Verify final outputs are only -1 or 1
    unique_outputs = torch.unique(output)
    print(f"Unique output values: {unique_outputs.detach().numpy()}")
    assert torch.all(torch.abs(output) == 1), "All outputs should have magnitude 1"
    print("BinaryFCNN test passed!")

def main():
    print("Testing Binary Neural Network components")
    test_binary_linear()
    test_sign_activation()
    test_binary_fcnn()
    print("\nAll tests passed! The Binary Neural Network is working correctly.")

if __name__ == "__main__":
    main()