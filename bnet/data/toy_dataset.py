import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import numpy as np

def generate_toy_data(config):
    """
    Generates a toy dataset based on the configuration.

    Args:
        config (dict): A dictionary containing dataset parameters.

    Returns:
        tuple: A tuple containing training and testing data as PyTorch Tensors:
               ((X_train, y_train), (X_test, y_test)).
    """
    dataset_type = config['dataset']['dataset_type']
    num_samples = config['dataset']['num_samples']
    num_features = config['dataset']['num_features']
    noise = config['dataset']['noise']
    random_state = config['dataset']['random_state']
    test_split_ratio = config['dataset']['test_split_ratio']

    if dataset_type == 'classification':
        num_classes = config['dataset']['num_classes']
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=max(2, num_features // 2),
            n_redundant=max(0, num_features // 4),
            n_classes=num_classes,
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=random_state
        )
        # Reshape y to be (n_samples, 1) for BCEWithLogitsLoss
        if config['training']['loss_function'] == 'BCEWithLogitsLoss':
            y = y.astype(np.float32).reshape(-1, 1)
        else:
            y = y.astype(np.int64) # For CrossEntropyLoss

    elif dataset_type == 'regression':
        X, y = make_regression(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=max(2, num_features // 2),
            noise=noise,
            random_state=random_state
        )
        y = y.astype(np.float32).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_ratio, random_state=random_state
    )

    # Convert to PyTorch Tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor)

class ToyDataset(Dataset):
    """Standard PyTorch Dataset for the toy data."""
    def __init__(self, X, y):
        """
        Args:
            X (torch.Tensor): Features.
            y (torch.Tensor): Labels.
        """
        self.X = X
        self.y = y

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Returns a single sample."""
        return self.X[idx], self.y[idx]

def get_toy_dataloaders(config):
    """
    Generates data and creates PyTorch DataLoaders.

    Args:
        config (dict): A dictionary containing dataset and training parameters.

    Returns:
        tuple: A tuple containing the training and testing DataLoaders.
    """
    (X_train, y_train), (X_test, y_test) = generate_toy_data(config)

    train_dataset = ToyDataset(X_train, y_train)
    test_dataset = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    return train_loader, test_loader