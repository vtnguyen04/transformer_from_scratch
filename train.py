from jsonargparse import CLI
import lightning as L
from lightning.pytorch.callbacks import RichModelSummary
from transformer_basic.callbacks import CustomRichProgressBar

from transformer_basic.datamodules.reverse_string import ReverseStringDataModule
from transformer_basic.lit_module import LitTransformer
from transformer_basic.tokenizer import CharTokenizer


def train(
    model_config: dict,
    optimizer_config: dict,
    datamodule: ReverseStringDataModule,
    trainer: L.Trainer,
    tokenizer_path: str,
):
    """
    The main training function.

    Args:
        model_config: Dictionary with model parameters.
        optimizer_config: Dictionary with optimizer parameters.
        datamodule: The LightningDataModule for data handling.
        trainer: The PyTorch Lightning Trainer.
        tokenizer_path: Path to save the trained tokenizer.
    """
    # --- Rich Callbacks ---
    progress_bar = CustomRichProgressBar()
    # Use max_depth=2 to see more details of the Transformer architecture
    model_summary = RichModelSummary(max_depth=2)
    
    if trainer.callbacks:
        trainer.callbacks.extend([progress_bar, model_summary])
    else:
        trainer.callbacks = [progress_bar, model_summary]

    # --- Data Preparation & Tokenizer ---
    datamodule.prepare_data()
    tokenizer = datamodule.tokenizer
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer trained and saved to {tokenizer_path}")

    # --- Update Model Config ---
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.pad_id
    model_config["src_vocab_size"] = vocab_size
    model_config["tgt_vocab_size"] = vocab_size
    # Although pad_id is not directly used by Transformer, it's good practice
    # and used by the LitModule's loss function.
    model_config["pad_id"] = pad_id
    print(f"Model configured with vocab size: {vocab_size} and pad_id: {pad_id}")

    # --- Instantiate Lightning Module ---
    lit_module = LitTransformer(model_config=model_config, optimizer_config=optimizer_config)

    # --- Training ---
    trainer.fit(lit_module, datamodule=datamodule)


if __name__ == "__main__":
    cli = CLI(
        train,
        as_positional=False,
        default_config_files=["configs/base_config.yaml"],
    )
