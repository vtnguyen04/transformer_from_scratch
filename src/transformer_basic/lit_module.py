import torch
import torch.nn as nn
import lightning as L
import importlib
from .models.transformer import Transformer, generate_square_subsequent_mask

class LitTransformer(L.LightningModule):
    """
    PyTorch Lightning module for the Transformer model.
    """
    def __init__(self, model_config: dict, optimizer_config: dict):
        super().__init__()
        self.save_hyperparameters() # Saves model_config and optimizer_config to hparams
        
        model_args = {k: v for k, v in model_config.items() if k not in ["_target_", "pad_id"]}
        self.model = Transformer(**model_args)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.hparams.model_config.get("pad_id", 0))

    def _create_masks(self, src: torch.Tensor, tgt: torch.Tensor, pad_id: int):
        """
        Creates the necessary masks for the Transformer model.
        """
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        # Look-ahead mask for the decoder
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(self.device)
        
        # Padding masks
        src_padding_mask = (src == pad_id)
        tgt_padding_mask = (tgt == pad_id)
        
        return tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        # The target for the decoder input is everything except the last token
        tgt_input = batch["tgt"][:, :-1]
        
        # Create masks
        # Note: The 'pad_id' must be passed correctly during instantiation.
        # We will connect this in the train.py script.
        pad_id = self.hparams.model_config.get("pad_id", 0)
        tgt_mask, src_padding_mask, tgt_padding_mask = self._create_masks(
            batch["src"], tgt_input, pad_id
        )

        return self.model(
            src=batch["src"],
            tgt=tgt_input,
            src_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

    def _shared_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Logic for a single training or validation step.
        """
        logits = self(batch)
        
        # The target for the loss is everything except the first token (<SOS>)
        tgt_out = batch["tgt"][:, 1:]
        
        # Reshape for CrossEntropyLoss: (N, C, ...)
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def generate(self, src: torch.Tensor, sos_id: int, eos_id: int, max_length: int = 50):
        """
        Generates an output sequence for a given source sequence.
        This is an autoregressive decoding process.
        """
        self.model.eval()
        device = self.device
        src = src.to(device)

        pad_id = self.hparams.model_config.get("pad_id", 0)

        # Create source padding mask
        src_mask = (src == pad_id).unsqueeze(1).unsqueeze(2).to(device)

        with torch.no_grad():
            memory = self.model.encode(src, src_mask)
        
        # Initialize decoder input with start-of-sequence token
        ys = torch.full((1, 1), sos_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            with torch.no_grad():
                # Create a subsequent mask for the decoder
                tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
                
                # Get model output
                out = self.model.decode(ys, memory, tgt_mask)
                
                # Get the logits for the last predicted token
                last_word_logits = self.model.generator(out[:, -1])
                
                # Get the most likely token
                next_word = torch.argmax(last_word_logits, dim=1)
            
            # Append the new token to the sequence
            ys = torch.cat([ys, torch.full((1, 1), next_word.item(), dtype=torch.long, device=device)], dim=1)
            
            # If the end-of-sequence token is predicted, stop generating
            if next_word.item() == eos_id:
                break
        
        return ys

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Sets up the optimizer.
        """
        optimizer_target = self.hparams.optimizer_config["_target_"]
        module_name, class_name = optimizer_target.rsplit(".", 1)
        module = importlib.import_module(module_name)
        optimizer_class = getattr(module, class_name)

        # Filter out the '_target_' key before passing to the optimizer
        optimizer_params = {k: v for k, v in self.hparams.optimizer_config.items() if k != "_target_"}

        optimizer = optimizer_class(self.parameters(), **optimizer_params)
        return optimizer
