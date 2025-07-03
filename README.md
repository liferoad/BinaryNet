# BinaryNet

A minimal PyTorch project to test research ideas with a Fully Connected Neural Network (FCNN) and a synthetic toy dataset. The project includes both a standard neural network with ReLU activations and a binary neural network where weights and activations are constrained to -1 or 1 values.

## Project Structure

```
bnet/
├── configs/
│   ├── toy_config.yaml
│   └── binary_config.yaml
├── data/
│   └── toy_dataset.py
├── models/
│   ├── simple_fcnn.py
│   └── binary_fcnn.py
├── trainer/
│   └── basic_trainer.py
├── utils/
│   └── project_utils.py
├── main.py
├── main_binary.py
└── requirements.txt
tests/
└── test_binary_model.py
```

## Setup

This project requires **Python 3.12**.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd BinaryNet
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r bnet/requirements.txt
    ```

## Usage

### Standard Neural Network

To run the training process with the standard FCNN model, use the `main.py` script and specify the configuration file.

```bash
python bnet/main.py --config bnet/configs/toy_config.yaml
```

### Binary Neural Network

To run the training process with the binary neural network model (weights and activations constrained to -1 or 1), use the `main_binary.py` script:

```bash
python bnet/main_binary.py --config bnet/configs/binary_config.yaml
```

### Testing the Binary Neural Network

To run tests that verify the binary neural network is correctly constraining weights and activations to -1 or 1:

```bash
python -m tests.test_binary_model
```

### Configuration

You can modify the configuration files to change the experiment parameters:

* **Standard model:** `bnet/configs/toy_config.yaml`
* **Binary model:** `bnet/configs/binary_config.yaml`

Parameters you can adjust include:

*   **Dataset:** `dataset_type`, `num_samples`, `num_features`, etc.
*   **Model:** `input_size`, `hidden_sizes`, `output_size`.
*   **Training:** `epochs`, `batch_size`, `learning_rate`, etc.

### TensorBoard

Logs and model checkpoints are saved in the following directories:

* Standard model: `runs/`
* Binary model: `runs_binary/`

You can use TensorBoard to visualize the training process:

```bash
# For standard model
tensorboard --logdir runs

# For binary model
tensorboard --logdir runs_binary

# For both models
tensorboard --logdir_spec standard:runs,binary:runs_binary
```