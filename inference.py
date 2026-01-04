import torch
import argparse
from pathlib import Path
from transformer_basic.lit_module import LitTransformer
from transformer_basic.tokenizer import CharTokenizer

def infer(args):
    """
    Loads a trained model and performs inference on a given text.
    """
    # --- Load Tokenizer ---
    if not Path(args.tokenizer_path).exists():
        print(f"Error: Tokenizer file not found at {args.tokenizer_path}")
        print("Please ensure you have trained the model first, which saves the tokenizer.")
        return

    tokenizer = CharTokenizer.from_file(args.tokenizer_path)
    print(f"Tokenizer loaded from {args.tokenizer_path}")

    # --- Load Model ---
    if not Path(args.checkpoint_path).exists():
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        print("Please provide the correct path to a trained model checkpoint (.ckpt).")
        print("Checkpoints are usually saved in the 'lightning_logs' directory.")
        return

    # Load the Lightning module from the checkpoint
    model = LitTransformer.load_from_checkpoint(
        args.checkpoint_path,
        map_location=torch.device("cpu"), # Load on CPU
        strict=False
    )
    model.freeze() # Set to evaluation mode and disable gradients
    print(f"Model loaded from {args.checkpoint_path}")

    # --- Inference ---
    print(f"\nInput text: '{args.text}'")

    # Encode the source text
    src_tokens = tokenizer.encode(args.text, add_special_tokens=True)
    src_tensor = torch.tensor(src_tokens).unsqueeze(0) # Add batch dimension

    # Generate the output sequence
    generated_tokens = model.generate(
        src=src_tensor,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
        max_length=len(src_tokens) + 5, # Set a reasonable max length
    )

    # Decode the generated tokens into a string
    # .squeeze(0) removes the batch dimension
    result_text = tokenizer.decode(generated_tokens.squeeze(0).tolist(), skip_special_tokens=True)

    print(f"Model output: '{result_text}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for the Basic Transformer model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.ckpt file).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="hello",
        help="The input text to be reversed by the model.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="data/tokenizer.json",
        help="Path to the saved tokenizer file.",
    )
    
    args = parser.parse_args()
    infer(args)
