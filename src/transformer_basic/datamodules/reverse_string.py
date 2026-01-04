import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
from ..tokenizer import CharTokenizer
import random
import string

class ReverseStringDataset(Dataset):
    """
    A dataset for the string reversal task.
    Generates pairs of (string, reversed_string).
    """
    def __init__(self, data_pairs: list[tuple[str, str]], tokenizer: CharTokenizer):
        super().__init__()
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_text, tgt_text = self.data_pairs[idx]
        src_tokens = self.tokenizer.encode(src_text, add_special_tokens=True)
        tgt_tokens = self.tokenizer.encode(tgt_text, add_special_tokens=True)
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

class ReverseStringDataModule(L.LightningDataModule):
    """
    A LightningDataModule for the string reversal task.
    It generates data on the fly and prepares DataLoaders.
    """
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        num_samples: int = 10000,
        min_length: int = 5,
        max_length: int = 20,
        train_split: float = 0.8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.train_split = train_split
        self.tokenizer: CharTokenizer | None = None
        self.data_pairs: list[tuple[str, str]] = []

    def prepare_data(self) -> None:
        # This method is for downloading/generating data.
        # It's called on only one process in DDP.
        
        # 1. Generate random character sequences
        chars = string.ascii_lowercase + string.digits
        all_text = []
        for _ in range(self.num_samples):
            length = random.randint(self.min_length, self.max_length)
            text = "".join(random.choices(chars, k=length))
            all_text.append(text)
            self.data_pairs.append((text, text[::-1]))

        # 2. Train tokenizer
        self.tokenizer = CharTokenizer()
        self.tokenizer.train(iter(all_text))

    def setup(self, stage: str | None = None) -> None:
        # This method is for splitting data, creating datasets, etc.
        # It's called on every process in DDP.
        if not self.data_pairs:
             self.prepare_data() # Ensure data is generated if not already
        
        full_dataset = ReverseStringDataset(self.data_pairs, self.tokenizer)
        
        train_size = int(self.train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def _collate_fn(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Pads sequences in a batch to the same length.
        """
        pad_id = self.tokenizer.pad_id
        
        src_tensors, tgt_tensors = zip(*batch)
        
        src_padded = torch.nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=pad_id)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_id)
        
        return {"src": src_padded, "tgt": tgt_padded}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
