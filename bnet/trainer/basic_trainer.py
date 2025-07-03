import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import os
import numpy as np

class BasicTrainer:
    """A basic trainer for a PyTorch model."""

    def __init__(self, model, train_loader, test_loader, config):
        """
        Initializes the BasicTrainer.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): The data loader for the training set.
            test_loader (DataLoader): The data loader for the test set.
            config (dict): A dictionary containing training and general configurations.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device(config['general']['device'])
        self.model.to(self.device)

        # Optimizer
        self.optimizer = self._get_optimizer()

        # Loss function
        self.loss_function = self._get_loss_function()

        # TensorBoard SummaryWriter
        self.log_dir = os.path.join(config['general']['log_dir'], f"{config['dataset']['dataset_type']}_{config['training']['optimizer']}_{config['training']['loss_function']}_{config['general']['seed']}")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.best_metric = float('inf') if 'MSE' in str(self.loss_function) else float('-inf')

    def _get_optimizer(self):
        """Creates the optimizer based on the config."""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _get_loss_function(self):
        """Creates the loss function based on the config."""
        loss_name = self.config['training']['loss_function'].lower()
        if loss_name == 'bcewithlogitsloss':
            return nn.BCEWithLogitsLoss()
        elif loss_name == 'crossentropyloss':
            return nn.CrossEntropyLoss()
        elif loss_name == 'mseloss':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def train_epoch(self, epoch):
        """Trains the model for one epoch."""
        self.model.train()  # Set the model to training mode
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} [Train]")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        return avg_loss

    def evaluate(self, epoch):
        """Evaluates the model on the test set."""
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        all_preds = []
        all_labels = []
        progress_bar = tqdm(self.test_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} [Eval]")

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                total_loss += loss.item()

                if self.config['dataset']['dataset_type'] == 'classification':
                    if 'BCE' in str(self.loss_function):
                        preds = torch.sigmoid(outputs) > 0.5
                    else: # CrossEntropy
                        preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.test_loader)
        self.writer.add_scalar('Loss/test', avg_loss, epoch)

        metrics = {'loss': avg_loss}
        if self.config['dataset']['dataset_type'] == 'classification':
            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            self.writer.add_scalar('Accuracy/test', accuracy * 100, epoch)
            metrics['accuracy'] = accuracy

        return metrics

    def train(self):
        """The main training loop."""
        for epoch in range(self.config['training']['epochs']):
            self.train_epoch(epoch)
            metrics = self.evaluate(epoch)

            # Checkpointing
            metric_to_check = metrics['loss'] if 'MSE' in str(self.loss_function) else metrics.get('accuracy', 0)
            is_better = (metric_to_check < self.best_metric) if 'MSE' in str(self.loss_function) else (metric_to_check > self.best_metric)

            if is_better:
                self.best_metric = metric_to_check
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_model.pth'))
                print(f"New best model saved at epoch {epoch+1} with metric: {self.best_metric:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, f'checkpoint_epoch_{epoch+1}.pth'))

            # Print test loss and accuracy in the same style as BinaryTrainer
            if self.config['dataset']['dataset_type'] == 'classification':
                accuracy = metrics.get('accuracy', 0) * 100  # convert to percent
                print(f"Epoch [{epoch+1}/{self.config['training']['epochs']}], Test Loss: {metrics['loss']:.4f}, Test Accuracy: {accuracy:.2f}%")
            else:
                print(f"Epoch [{epoch+1}/{self.config['training']['epochs']}], Test Loss: {metrics['loss']:.4f}")

        self.writer.close()
        print("Training finished.")