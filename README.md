# BinaryNet

A minimal PyTorch project to test research ideas with a Fully Connected Neural Network (FCNN) and a synthetic toy dataset.

## Project Structure

```
bnet/
├── configs/
│   └── toy_config.yaml
├── data/
│   └── toy_dataset.py
├── models/
│   └── simple_fcnn.py
├── trainer/
│   └── basic_trainer.py
├── utils/
│   └── project_utils.py
├── main.py
└── requirements.txt
```

## Setup

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

To run the training process, use the `main.py` script and specify the configuration file.

```bash
python bnet/main.py --config bnet/configs/toy_config.yaml
```

### Configuration

You can modify the `bnet/configs/toy_config.yaml` file to change the experiment parameters, such as:

*   **Dataset:** `dataset_type`, `num_samples`, `num_features`, etc.
*   **Model:** `input_size`, `hidden_sizes`, `output_size`.
*   **Training:** `epochs`, `batch_size`, `learning_rate`, etc.

### TensorBoard

Logs and model checkpoints are saved in the `runs/` directory. You can use TensorBoard to visualize the training process:

```bash
tensorboard --logdir runs
```