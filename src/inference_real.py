"""
Inference script for generating captions with trained OpenVision Student on real data.

This script:
- Loads a trained model checkpoint
- Loads the tokenizer
- Generates captions for images
- Displays both token IDs and decoded text
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

from model import OpenVisionStudent
from tokenizer import SimpleTokenizer


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    """
    Load trained model and tokenizer from checkpoints.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        tokenizer_path: Path to tokenizer file (.json file)
        device: Device to load model on

    Returns:
        model: Loaded OpenVisionStudent model
        tokenizer: Loaded SimpleTokenizer
        checkpoint: Full checkpoint dict (contains metadata)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same architecture
    model = OpenVisionStudent(
        vocab_size=checkpoint['vocab_size'],
        text_dim=checkpoint['text_dim'],
        num_heads=checkpoint['num_heads'],
        num_decoder_layers=checkpoint['num_decoder_layers'],
        max_seq_len=checkpoint['max_seq_len'],
    )

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    return model, tokenizer, checkpoint


def preprocess_image(image_path):
    """
    Load and preprocess an image for the model.

    Args:
        image_path: Path to image file

    Returns:
        image_tensor: Preprocessed image tensor (1, 3, 224, 224)
    """
    # Same preprocessing as training (ViT standard)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor


def generate_caption(model, image, tokenizer, max_length=50, temperature=1.0, device='cpu'):
    """
    Generate caption for an image using greedy decoding.

    Args:
        model: Trained OpenVisionStudent model
        image: Preprocessed image tensor (1, 3, 224, 224)
        tokenizer: SimpleTokenizer instance
        max_length: Maximum caption length
        temperature: Sampling temperature (1.0 = greedy, >1.0 = more random)
        device: Device to run on

    Returns:
        List of token IDs representing the generated caption
    """
    model.eval()
    image = image.to(device)

    # Encode image once (reuse for all decoding steps)
    with torch.no_grad():
        visual_tokens = model.visual_encoder(image)  # (1, 197, 768)
        visual_tokens = model.projector(visual_tokens)  # (1, 197, 512)

    # Start with START token
    caption_tokens = [tokenizer.START_ID]

    # Generate tokens one by one
    with torch.no_grad():
        for _ in range(max_length - 1):
            # Convert current tokens to tensor
            current_caption = torch.tensor([caption_tokens], dtype=torch.long, device=device)

            # Get decoder output (logits)
            output = model.decoder(
                text_tokens=current_caption,
                visual_tokens=visual_tokens,
            )

            # Get logits for the last position (next token prediction)
            next_token_logits = output[0, -1, :]  # (vocab_size,)

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Greedy decoding: take most likely token
            next_token = torch.argmax(next_token_logits).item()

            # Add to caption
            caption_tokens.append(next_token)

            # Stop if END token generated
            if next_token == tokenizer.END_ID:
                break

    return caption_tokens


def main(args):
    """Main inference function"""

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    print("=" * 70)

    # Load model and tokenizer
    print(f"Loading model from: {args.checkpoint}")
    print(f"Loading tokenizer from: {args.tokenizer}")

    model, tokenizer, checkpoint = load_model_and_tokenizer(
        args.checkpoint,
        args.tokenizer,
        device
    )

    print(f"\nModel info:")
    print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
    print(f"  Best validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Vocabulary size: {checkpoint['vocab_size']}")
    print("=" * 70)

    # Load and preprocess image
    print(f"\nGenerating caption for: {args.image}")
    image_tensor = preprocess_image(args.image)

    # Generate caption
    print(f"Generating with temperature={args.temperature}...")
    caption_tokens = generate_caption(
        model,
        image_tensor,
        tokenizer,
        max_length=args.max_length,
        temperature=args.temperature,
        device=device
    )

    # Decode to text
    caption_text = tokenizer.decode(caption_tokens, skip_special_tokens=True)

    # Display results
    print("\n" + "=" * 70)
    print("GENERATED CAPTION")
    print("=" * 70)
    print(f"\nTokens: {caption_tokens}")
    print(f"\nCaption: {caption_text}")
    print("\n" + "=" * 70)

    # If multiple temperatures requested, try them
    if args.show_variants:
        print("\nTrying different temperatures:")
        print("-" * 70)
        for temp in [0.7, 1.0, 1.3]:
            caption_tokens = generate_caption(
                model, image_tensor, tokenizer,
                max_length=args.max_length,
                temperature=temp,
                device=device
            )
            caption_text = tokenizer.decode(caption_tokens, skip_special_tokens=True)
            print(f"Temperature {temp}: {caption_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for images")

    # Required arguments
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_real.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json",
                        help="Path to tokenizer file")
    parser.add_argument("--image", type=str, default="data/Images/1000268201_693b08cb0e.jpg",
                        help="Path to image file")

    # Generation arguments
    parser.add_argument("--max-length", type=int, default=50,
                        help="Maximum caption length (default: 50)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--show-variants", action="store_true",
                        help="Show captions with different temperatures")

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Please train the model first using: python src/train_real.py --save-model")
        exit(1)

    if not Path(args.tokenizer).exists():
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        print("Tokenizer should be saved alongside the model checkpoint.")
        exit(1)

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        exit(1)

    main(args)
