import json
from pathlib import Path
from typing import Iterator

class CharTokenizer:
    """
    A simple character-level tokenizer.
    It learns a vocabulary from the data and can encode/decode strings.
    """
    def __init__(self, special_tokens: list[str] | None = None):
        if special_tokens is None:
            # PAD: for padding sequences to the same length
            # SOS: Start Of Sequence
            # EOS: End Of Sequence
            # UNK: for unknown characters
            self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        else:
            self.special_tokens = special_tokens

        self.vocab = self.special_tokens.copy()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def sos_id(self) -> int:
        return self.token_to_id["<SOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    def train(self, text_iterator: Iterator[str]) -> None:
        """
        Builds the vocabulary from an iterator of strings.
        """
        char_set = set()
        for text in text_iterator:
            char_set.update(list(text))

        for char in sorted(list(char_set)):
            if char not in self.token_to_id:
                self.vocab.append(char)
                new_id = len(self.vocab) - 1
                self.token_to_id[char] = new_id
                self.id_to_token[new_id] = char

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encodes a string into a list of token IDs.
        """
        encoded = [self.token_to_id.get(char, self.token_to_id["<UNK>"]) for char in text]
        if add_special_tokens:
            return [self.sos_id] + encoded + [self.eos_id]
        return encoded

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        chars = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id)
            if skip_special_tokens and token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, file_path: str | Path) -> None:
        """
        Saves the tokenizer's vocabulary to a file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            save_data = {
                "vocab": self.vocab,
                "special_tokens": self.special_tokens
            }
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "CharTokenizer":
        """
        Loads a tokenizer from a saved vocabulary file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            save_data = json.load(f)
        
        tokenizer = cls(special_tokens=save_data["special_tokens"])
        tokenizer.vocab = save_data["vocab"]
        tokenizer.token_to_id = {token: i for i, token in enumerate(tokenizer.vocab)}
        tokenizer.id_to_token = {i: token for i, token in enumerate(tokenizer.vocab)}
        
        return tokenizer
