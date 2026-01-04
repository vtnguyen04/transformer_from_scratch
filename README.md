# Basic Transformer Implementation

A from-scratch implementation of the original Transformer model ("Attention Is All You Need") using PyTorch Lightning.

This project is designed to be a clear, well-structured, and reusable foundation for sequence-to-sequence tasks.

## Project Structure

```
transformer_basic/
├── configs/
│   └── base_config.yaml      # Configuration for training
├── data/
│   └── README.md             # Instructions for data placement (synthetic data generated)
├── src/
│   └── transformer_basic/
│       ├── models/           # Core Transformer model components (MultiHeadAttention, Encoder, Decoder, Transformer)
│       ├── datamodules/      # PyTorch Lightning DataModules (e.g., ReverseStringDataModule)
│       ├── lit_module.py     # Main LightningModule for training logic and inference (`generate` method)
│       ├── tokenizer.py      # Simple character-level tokenizer
│       └── callbacks.py      # Custom callbacks (e.g., CustomRichProgressBar)
│       └── __init__.py
├── .gitignore
├── pyproject.toml            # Project metadata and dependencies
├── README.md
├── inference.py              # Script to run inference with a trained model
└── train.py                  # Main training script
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vtnguyen04/transformer_from_scratch.git
   cd transformer_basic
   ```

2. (Recommended) Create a virtual environment and install dependencies using `uv`:
   ```bash
   uv sync
   ```
   This will install all required packages and set up the project in editable mode.

## Usage

### Training the Model

To start training the model with the default configuration:

```bash
uv run train.py
```
Or, to explicitly use the base config:
```bash
uv run train.py --default_config_files configs/base_config.yaml
```
You can override any configuration parameter directly from the command line (e.g., `--trainer.max_epochs 5`).
*   **Config-Driven Scalability:** Easily swap out models, optimizers, or data modules by simply updating the configuration file (`configs/base_config.yaml`) or passing arguments via the CLI, without modifying `train.py`.

### Running Inference

After training, you can use the `inference.py` script to get predictions from your trained model.

1.  **Locate a Checkpoint:** Find the path to your desired model checkpoint. Checkpoints are typically saved in the `lightning_logs/version_X/checkpoints/` directory. For example: `lightning_logs/version_7/checkpoints/epoch=0-step=500.ckpt`
2.  **Run Inference:**
    ```bash
    uv run inference.py \
      --checkpoint_path lightning_logs/version_9/checkpoints/epoch=9-step=1250.ckpt \
      --text "hello"
    ```
    Replace `lightning_logs/version_7/checkpoints/epoch=0-step=500.ckpt` with the actual path to your checkpoint and `"hello"` with your desired input text.

The model will output the reversed string. For instance, input "transformer" should yield "remorfsart".
