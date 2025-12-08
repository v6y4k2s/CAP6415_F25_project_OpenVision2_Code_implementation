"""
Demo script for generating captions for 6 diverse images.
This script is designed for video demonstration purposes.

Usage:
    python demo_6_images.py --checkpoint checkpoints/best_model_real.pt --tokenizer checkpoints/tokenizer.json
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.model import OpenVisionStudent
from src.tokenizer import SimpleTokenizer


# 6 diverse images for demonstration (manually selected for variety)
DEMO_IMAGES = [
    {
        "path": "data/Images/1000268201_693b08cb0e.jpg",
        "description": "Child in pink dress",
        "category": "Children/People"
    },
    {
        "path": "data/Images/1001773457_577c3a7d70.jpg",
        "description": "Two dogs interacting",
        "category": "Animals"
    },
    {
        "path": "data/Images/1002674143_1b742ab4b8.jpg",
        "description": "Girl painting rainbow",
        "category": "Activities/Art"
    },
    {
        "path": "data/Images/1003163366_44323f5815.jpg",
        "description": "Man on bench with dog",
        "category": "People + Animals"
    },
    {
        "path": "data/Images/1007129816_e794419615.jpg",
        "description": "Man with orange hat",
        "category": "Portrait"
    },
    {
        "path": "data/Images/1007320043_627395c3d8.jpg",
        "description": "Child climbing rope",
        "category": "Action/Sports"
    },
]


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    """Load trained model and tokenizer from checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = OpenVisionStudent(
        vocab_size=checkpoint['vocab_size'],
        text_dim=checkpoint['text_dim'],
        num_heads=checkpoint['num_heads'],
        num_decoder_layers=checkpoint['num_decoder_layers'],
        max_seq_len=checkpoint['max_seq_len'],
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    tokenizer = SimpleTokenizer.load(tokenizer_path)

    return model, tokenizer, checkpoint


def preprocess_image(image_path, device):
    """Preprocess image for inference (same as training)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)


def generate_caption(model, image_tensor, tokenizer, max_length=50, temperature=1.0):
    """Generate caption for a single image using greedy decoding."""
    model.eval()

    with torch.no_grad():
        # Encode image once
        visual_tokens = model.visual_encoder(image_tensor)
        visual_tokens = model.projector(visual_tokens)

        # Start with START token
        caption = [tokenizer.START_ID]

        for _ in range(max_length):
            # Create caption tensor
            caption_tensor = torch.tensor([caption], device=image_tensor.device)

            # Forward pass through decoder
            logits = model.decoder(caption_tensor, visual_tokens)

            # Get next token prediction (last position)
            next_token_logits = logits[0, -1, :] / temperature
            next_token = torch.argmax(next_token_logits).item()

            # Add to caption
            caption.append(next_token)

            # Stop if END token
            if next_token == tokenizer.END_ID:
                break

        return caption


def main():
    parser = argparse.ArgumentParser(description="Demo: Generate captions for 6 diverse images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer JSON file")
    parser.add_argument("--max-length", type=int, default=50,
                        help="Maximum caption length (default: 50)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0 = greedy)")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")
    print("=" * 80)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer, checkpoint = load_model_and_tokenizer(
        args.checkpoint, args.tokenizer, device
    )

    print(f"✓ Model loaded (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")
    print(f"✓ Vocabulary size: {checkpoint['vocab_size']} words")
    print("=" * 80)
    print()

    # Generate captions for all 6 demo images
    results = []

    for i, image_info in enumerate(DEMO_IMAGES, 1):
        image_path = image_info["path"]

        # Check if image exists
        if not Path(image_path).exists():
            print(f"⚠️  Image {i}: {image_path} not found, skipping...")
            continue

        print(f"\n{'=' * 80}")
        print(f"Image {i}/6: {image_info['description']}")
        print(f"Category: {image_info['category']}")
        print(f"Path: {image_path}")
        print("-" * 80)

        # Preprocess image
        image_tensor = preprocess_image(image_path, device)

        # Generate caption
        caption_tokens = generate_caption(
            model, image_tensor, tokenizer,
            max_length=args.max_length,
            temperature=args.temperature
        )

        # Decode to text
        caption_text = tokenizer.decode(caption_tokens, skip_special_tokens=True)

        # Display results
        print(f"Generated Caption:")
        print(f"  → \"{caption_text}\"")
        print(f"\nToken IDs: {caption_tokens[:15]}{'...' if len(caption_tokens) > 15 else ''}")

        # Store results
        results.append({
            "image": image_path,
            "description": image_info["description"],
            "category": image_info["category"],
            "caption": caption_text,
            "tokens": caption_tokens
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - All Generated Captions")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['description']} ({result['category']})")
        print(f"   Caption: \"{result['caption']}\"")

    print("\n" + "=" * 80)
    print(f"✓ Successfully generated captions for {len(results)}/6 images")
    print("=" * 80)


if __name__ == "__main__":
    main()
