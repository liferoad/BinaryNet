import argparse
import os
import yaml

from utils.project_utils import set_seed, load_config, create_dirs
from data.toy_dataset import get_toy_dataloaders
from models.simple_fcnn import build_model
from trainer.basic_trainer import BasicTrainer

def main():
    """Main function to run the experiment."""
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="A simple PyTorch FCNN toy project.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    # 2. Load configuration
    config = load_config(args.config)

    # 3. Set random seed for reproducibility
    set_seed(config['general']['seed'])

    # 4. Create directories for logging and checkpoints
    log_dir_base = config['general']['log_dir']
    run_name = f"{config['dataset']['dataset_type']}_{config['training']['optimizer']}_{config['training']['loss_function']}_{config['general']['seed']}"
    log_dir = os.path.join(log_dir_base, run_name)
    create_dirs([log_dir])

    # 5. Save a copy of the config to the log directory for this run
    config_save_path = os.path.join(log_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    # 6. Get data loaders
    print("Loading data...")
    train_loader, test_loader = get_toy_dataloaders(config)
    print("Data loaded.")

    # 7. Build the model
    print("Building model...")
    model = build_model(config)
    print(model)
    print("Model built.")

    # 8. Initialize and run the trainer
    print("Initializing trainer...")
    trainer = BasicTrainer(model, train_loader, test_loader, config)
    print("Trainer initialized. Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()