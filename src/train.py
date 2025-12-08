"""
Training script for OpenVision Student

This script implements a complete training loop with:
- Training and validation phases
- Loss tracking and logging
- Model checkpointing
- Progress visualization
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from model import OpenVisionStudent
from dataset import create_dataloaders


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
    model.train()  # Set to training mode (enables dropout, batch norm updates, etc.)
    total_loss = 0.0
    num_batches = len(dataloader)

    # Progress bar for this epoch
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        captions = batch['caption'].to(device)

        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass: compute loss
        # apply_visual_masking=True during training for robustness
        loss = model.compute_loss(images, captions, apply_visual_masking=True)

        # Backward pass: compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Update progress bar
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
    model.eval()  # Set to evaluation mode (disables dropout, batch norm in eval mode)
    total_loss = 0.0
    num_batches = len(dataloader)

    # Progress bar for validation
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]  ")

    with torch.no_grad():  # Disable gradient computation for validation
        for batch in pbar:
            images = batch['image'].to(device)
            captions = batch['caption'].to(device)

            # Forward pass only (no masking during validation)
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

    print("-" * 60)

    # Create dataloaders
    print(f"Creating dataloaders...")
    print(f"  Train samples: {args.train_samples}")
    print(f"  Val samples: {args.val_samples}")
    print(f"  Batch size: {args.batch_size}")

    train_loader, val_loader = create_dataloaders(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        caption_length=args.caption_length,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print("-" * 60)

    # Create model
    print(f"Creating model...")
    model = OpenVisionStudent(
        vocab_size=args.vocab_size,
        text_dim=args.text_dim,
        num_heads=args.num_heads,
        num_decoder_layers=args.num_decoder_layers,
        max_seq_len=args.max_seq_len,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("-" * 60)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"Starting training for {args.epochs} epochs...\n")

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
                save_path = Path(args.output_dir) / "best_model.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"   Saved best model to {save_path}")

        print("-" * 60)

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")

    # Show loss progression
    print("\nLoss progression:")
    print("Epoch | Train Loss | Val Loss")
    print("------|------------|----------")
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        marker = " <-- best" if vl == best_val_loss else ""
        print(f"{i+1:5d} | {tl:10.4f} | {vl:8.4f}{marker}")

    if args.save_model:
        print(f"\nModel saved to: {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OpenVision Student model")

    # Data parameters
    parser.add_argument("--train-samples", type=int, default=80,
                        help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=20,
                        help="Number of validation samples")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--vocab-size", type=int, default=10000,
                        help="Vocabulary size")
    parser.add_argument("--caption-length", type=int, default=20,
                        help="Maximum caption length")

    # Model parameters
    parser.add_argument("--text-dim", type=int, default=512,
                        help="Text/decoder model dimension")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num-decoder-layers", type=int, default=6,
                        help="Number of decoder layers")
    parser.add_argument("--max-seq-len", type=int, default=128,
                        help="Maximum sequence length")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output parameters
    parser.add_argument("--save-model", action="store_true",
                        help="Save the best model")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")

    args = parser.parse_args()
    main(args)
