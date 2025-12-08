"""
Caption Generation (Inference) Module

Implements greedy decoding: at each step, select the word with 
highest probability.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional

# Special Token IDs (must match your tokenizer/vocabulary)
PAD_TOKEN_ID = 0    # Padding token
START_TOKEN_ID = 1  # Start of caption token
END_TOKEN_ID = 2    # End of caption token


def generate_caption(
    model,
    image: torch.Tensor,
    max_length: int = 20,
    device: str = 'cpu',
    start_token_id: int = START_TOKEN_ID,
    end_token_id: int = END_TOKEN_ID,
    temperature: float = 1.0,
) -> List[int]:
    """
    Generate a caption for a single image using greedy decoding.
    """
    model.eval()

    # Ensure image has batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    # Initialize caption
    caption_tokens = [start_token_id]

    with torch.no_grad():
        # Generate tokens one by one
        for _ in range(max_length):
            current_tokens = torch.tensor([caption_tokens], device=device)

            logits = model.forward(
                images=image,
                text_tokens=current_tokens,
                apply_masking=False  # No masking during inference
            )

            next_token_logits = logits[0, -1, :]
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)

            next_token_id = torch.argmax(probs).item()
            caption_tokens.append(next_token_id)

            if next_token_id == end_token_id:
                break

    return caption_tokens


def generate_caption_batch(
    model,
    images: torch.Tensor,
    max_length: int = 20,
    device: str = 'cpu',
    start_token_id: int = START_TOKEN_ID,
    end_token_id: int = END_TOKEN_ID,
) -> List[List[int]]:
    """
    Generate captions for a batch of images.
    """
    captions = []
    for i in range(images.shape[0]):
        caption = generate_caption(
            model=model,
            image=images[i],
            max_length=max_length,
            device=device,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
        )
        captions.append(caption)
    return captions


def tokens_to_text(token_ids: List[int], vocab: Optional[dict] = None) -> str:
    """
    Convert token IDs to text.
    """
    if vocab is None:
        return f"Token IDs: {token_ids}"

    words = []
    for token_id in token_ids:
        if token_id == START_TOKEN_ID:
            continue
        elif token_id == END_TOKEN_ID:
            break
        elif token_id == PAD_TOKEN_ID:
            continue
        else:
            words.append(vocab.get(token_id, f"<UNK_{token_id}>"))

    return " ".join(words)


# Demo/Testing code
if __name__ == "__main__":
    print("=" * 60)
    print("Caption Generation Demo")
    print("=" * 60)

    from model import OpenVisionStudent
    import os

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"\nError: No trained model found at {checkpoint_path}")
        print("Please train a model first using: python src/train.py --save-model")
        exit(1)

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model using checkpoint config
    model = OpenVisionStudent(
        vocab_size=checkpoint.get('vocab_size', 10000),
        visual_dim=768,
        text_dim=512,
        num_decoder_layers=6,
        num_heads=8,
        max_seq_len=128,  # Use default from training
        freeze_visual_encoder=True
    ).to(device)

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(
        f"Model loaded! (Epoch {checkpoint['epoch']}, "
        f"Val Loss: {checkpoint['val_loss']:.4f})"
    )

    print("\n" + "=" * 60)
    print("Generating caption for test image...")
    print("=" * 60)

    test_image = torch.randn(3, 224, 224)

    caption_tokens = generate_caption(
        model=model,
        image=test_image,
        max_length=20,
        device=device,
        temperature=1.0
    )

    print("\nGenerated caption (token IDs):")
    print(caption_tokens)
    print(f"\nCaption length: {len(caption_tokens)} tokens")

    print("\nToken breakdown:")
    for i, token_id in enumerate(caption_tokens):
        if token_id == START_TOKEN_ID:
            print(f"  {i}: {token_id} -> [START]")
        elif token_id == END_TOKEN_ID:
            print(f"  {i}: {token_id} -> [END]")
        elif token_id == PAD_TOKEN_ID:
            print(f"  {i}: {token_id} -> [PAD]")
        else:
            print(f"  {i}: {token_id} -> Word ID {token_id}")

    print("\n" + "=" * 60)
    print("Testing different temperatures (sampling randomness):")
    print("=" * 60)

    for temp in [0.8, 1.0, 1.5]:
        caption = generate_caption(
            model=model,
            image=test_image,
            max_length=20,
            device=device,
            temperature=temp
        )
        print(f"  Temperature {temp}: {caption}")

    print("\n" + "=" * 60)
    print("Note: We're using fake data, so captions are just token IDs.")
    print("Once we have a real tokenizer, these will be actual words!")
    print("=" * 60)
