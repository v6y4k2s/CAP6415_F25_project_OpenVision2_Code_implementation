"""
Training script for OpenVision Student on REAL Flickr8k data

This script:
- Loads real images and captions from Flickr8k dataset
- Builds vocabulary from captions
- Trains the model end-to-end
- Saves both model and tokenizer for inference
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from model import OpenVisionStudent
from real_dataset import create_flickr8k_dataloaders


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train the model for one epoch.

    Args:
        model: The OpenVisionStudent model
        dataloader: Training data loader
        optimizer: PyTorch optimizer
        device: Device to train on (cpu/cuda/mps)
        epoch: Current epoch number

    Returns:
        float: Average training loss for this epoch
    """
    model.train()  # Set to training mode
    total_loss = 0.0
    num_batches = len(dataloader)

    # Progress bar for this epoch
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for images, captions in pbar:
        # Move data to device
        images = images.to(device)
        captions = captions.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with visual masking (50% patches masked)
        loss = model.compute_loss(images, captions, apply_visual_masking=True)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, device, epoch):
    """
    Validate the model.

    Args:
        model: The OpenVisionStudent model
        dataloader: Validation data loader
        device: Device to run on
        epoch: Current epoch number

    Returns:
        float: Average validation loss
    """
    model.eval()  # Set to evaluation mode
    total_loss = 0.0
    num_batches = len(dataloader)

    # Progress bar for validation
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]  ")

    with torch.no_grad():  # No gradients needed for validation
        for images, captions in pbar:
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass WITHOUT visual masking (use all 197 tokens)
            loss = model.compute_loss(images, captions, apply_visual_masking=False)

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def main(args):
    """Main training function"""

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    print("=" * 70)
    print("TRAINING OPENVISION STUDENT ON FLICKR8K")
    print("=" * 70)

    # Create dataloaders and tokenizer
    print("\nLoading Flickr8k dataset...")
    train_loader, val_loader, test_loader, tokenizer = create_flickr8k_dataloaders(
        data_dir=args.data_dir,
        captions_file=args.captions_file,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        max_caption_length=args.max_caption_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
    )

    # Get actual vocab size (may be less than requested)
    actual_vocab_size = len(tokenizer)
    print(f"\nActual vocabulary size: {actual_vocab_size}")
    print("=" * 70)

    # Create model
    print("\nCreating model...")
    model = OpenVisionStudent(
        vocab_size=actual_vocab_size,
        text_dim=args.text_dim,
        num_heads=args.num_heads,
        num_decoder_layers=args.num_decoder_layers,
        max_seq_len=args.max_caption_length,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print("=" * 70)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70 + "\n")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, device, epoch)
        val_losses.append(val_loss)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save_model:
                # Create output directory
                output_dir = Path(args.output_dir)
                output_dir.mkdir(exist_ok=True)

                # Save checkpoint with both model and tokenizer
                checkpoint_path = output_dir / "best_model_real.pt"
                tokenizer_path = output_dir / "tokenizer.json"

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'vocab_size': actual_vocab_size,
                    'text_dim': args.text_dim,
                    'num_heads': args.num_heads,
                    'num_decoder_layers': args.num_decoder_layers,
                    'max_seq_len': args.max_caption_length,
                }, checkpoint_path)

                # Save tokenizer separately
                tokenizer.save(str(tokenizer_path))

                print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
                print(f"    Model: {checkpoint_path}")
                print(f"    Tokenizer: {tokenizer_path}")

        print("-" * 70)

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"\nLoss progression:")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12}")
    print("-" * 35)
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        best_marker = " ← best" if val_loss == best_val_loss else ""
        print(f"{i+1:<8} {train_loss:<12.4f} {val_loss:<12.4f}{best_marker}")

    if args.save_model:
        print(f"\n✓ Model and tokenizer saved to: {args.output_dir}/")
        print("\nTo generate captions, use:")
        print(f"  python src/inference_real.py --checkpoint {args.output_dir}/best_model_real.pt")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpenVision Student on Flickr8k")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to data directory (default: data)")
    parser.add_argument("--captions-file", type=str, default="captions.txt",
                        help="Name of captions file (default: captions.txt)")

    # Dataset split arguments
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")

    # Tokenizer arguments
    parser.add_argument("--vocab-size", type=int, default=5000,
                        help="Maximum vocabulary size (default: 5000)")
    parser.add_argument("--min-freq", type=int, default=2,
                        help="Minimum word frequency to include (default: 2)")
    parser.add_argument("--max-caption-length", type=int, default=50,
                        help="Maximum caption length (default: 50)")

    # Model architecture arguments
    parser.add_argument("--text-dim", type=int, default=512,
                        help="Text embedding dimension (default: 512)")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--num-decoder-layers", type=int, default=6,
                        help="Number of decoder layers (default: 6)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers (default: 4)")

    # Output arguments
    parser.add_argument("--save-model", action="store_true",
                        help="Save the best model checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()
    main(args)
