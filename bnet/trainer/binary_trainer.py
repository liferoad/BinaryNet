import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

class BinaryTrainer:
    """Trainer for binary neural networks."""
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.writer = SummaryWriter(log_dir=config['general']['log_dir'])

        # Optimizer and loss function
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()

    def _create_optimizer(self):
        optimizer_name = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        if optimizer_name.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_criterion(self):
        loss_name = self.config['training']['loss_function']
        if loss_name.lower() == 'bcewithlogitsloss':
            return nn.BCEWithLogitsLoss()
        elif loss_name.lower() == 'mseloss':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def train(self):
        num_epochs = self.config['training']['epochs']
        for epoch in range(num_epochs):
            self._train_epoch(epoch)
            self._evaluate(epoch)
        self.writer.close()

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} [Train]")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            # Adjust targets to be -1 and 1 for MSELoss
            adjusted_targets = targets.float() * 2 - 1
            if len(adjusted_targets.shape) == 1:
                adjusted_targets = adjusted_targets.unsqueeze(1)
            loss = self.criterion(outputs, adjusted_targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)

    def _evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        progress_bar = tqdm(self.test_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} [Eval]")
        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                # Adjust targets to be -1 and 1 for MSELoss
                adjusted_targets = targets.float() * 2 - 1
                if len(adjusted_targets.shape) == 1:
                    adjusted_targets = adjusted_targets.unsqueeze(1)
                loss = self.criterion(outputs, adjusted_targets)
                total_loss += loss.item()

                # Calculate accuracy
                predicted = (outputs > 0).float()
                if len(predicted.shape) > 1 and predicted.shape[1] == 1:
                    predicted = predicted.squeeze(1)
                accuracy = (predicted == targets.float()).float().mean()
                total_accuracy += accuracy.item()

        avg_loss = total_loss / len(self.test_loader)
        avg_accuracy = 100 * total_accuracy / len(self.test_loader)

        
        self.writer.add_scalar('Loss/test', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/test', avg_accuracy, epoch)
        print(f'Epoch [{epoch+1}/{self.config["training"]["epochs"]}], Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.2f}%')